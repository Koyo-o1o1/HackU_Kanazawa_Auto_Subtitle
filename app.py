# app.py
import os, uuid, re
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, abort, Response, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from tasks import process_video_task
from flask_httpauth import HTTPBasicAuth

BASE_DIR   = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "processed"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTS = {"mp4", "mov", "m4v", "avi", "mkv", "webm"}

app = Flask(__name__)
auth = HTTPBasicAuth()
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB

# --- Basic認証の設定 ---
users = {
    "admin": "change-this-password" # ← 必ずもっと複雑なパスワードに変更してください
}

@auth.verify_password
def verify_password(username, password):
    if username in users and users[username] == password:
        return username

# すべてのリクエストの前に認証を要求する
# ただし、ステータス確認URLは認証不要にする
@app.before_request
def require_login():
    if request.endpoint and 'static' not in request.endpoint and request.endpoint != 'task_status':
        return auth.login_required(lambda: None)()

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

def _partial_response(path: Path, mime: str):
    file_size = path.stat().st_size
    range_header = request.headers.get("Range")
    if range_header:
        m = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else file_size - 1
            start = max(0, start); end = min(end, file_size - 1)
            length = end - start + 1
            with open(path, "rb") as f:
                f.seek(start); data = f.read(length)
            rv = Response(data, 206, mimetype=mime, direct_passthrough=True)
            rv.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            rv.headers["Accept-Ranges"]  = "bytes"
            rv.headers["Content-Length"] = str(length)
            return rv
    with open(path, "rb") as f:
        data = f.read()
    rv = Response(data, 200, mimetype=mime, direct_passthrough=True)
    rv.headers["Accept-Ranges"]  = "bytes"
    rv.headers["Content-Length"] = str(file_size)
    return rv

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files: abort(400, "file フィールドがありません")
    f = request.files["file"]
    if f.filename == "":           abort(400, "ファイル名が空です")
    if not allowed_file(f.filename):
        abort(400, f"対応拡張子: {', '.join(sorted(ALLOWED_EXTS))}")

    uid = uuid.uuid4().hex
    in_ext = secure_filename(f.filename).rsplit(".", 1)[1].lower()
    in_path  = UPLOAD_DIR / f"{uid}.{in_ext}"
    
    f.save(in_path)

    raw_k = request.form.get("speakers") or request.form.get("k_speakers")
    try:
        k_speakers = int(raw_k)
        if k_speakers < 1: k_speakers = 1
        if k_speakers > 12: k_speakers = 12
    except (TypeError, ValueError):
        k_speakers = 2

    process_video_task.delay(uid, in_ext, k_speakers)
    
    return redirect(url_for("result", file_id=uid), code=303)


@app.route("/result/<file_id>")
def result(file_id):
    return render_template("result.html", file_id=file_id)

@app.route("/video/<file_id>")
def stream_video(file_id):
    p = OUTPUT_DIR / f"{file_id}_fs.mp4"
    if not p.exists():
        p = OUTPUT_DIR / f"{file_id}.mp4"
        if not p.exists():
            abort(404, "動画が見つかりません")
    return _partial_response(p, "video/mp4")

@app.route("/download/<file_id>")
def download(file_id):
    filename = f"{file_id}_fs.mp4" if (OUTPUT_DIR / f"{file_id}_fs.mp4").exists() else f"{file_id}.mp4"
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True, download_name="processed.mp4")

@app.route("/status/<file_id>")
def task_status(file_id):
    """タスクの完了状態を返すAPIエンドポイント"""
    # ★ 完了の合図である .done ファイルの存在を確認する
    done_file = OUTPUT_DIR / f"{file_id}.done"
    
    if done_file.exists():
        return jsonify({'status': 'complete'})
    else:
        return jsonify({'status': 'processing'})