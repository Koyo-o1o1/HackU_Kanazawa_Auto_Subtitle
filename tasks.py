# tasks.py
import os
import subprocess
from pathlib import Path
from celery import Celery
from main2 import process_video

celery_app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

BASE_DIR   = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "processed"

@celery_app.task
def process_video_task(uid: str, in_ext: str, k_speakers: int):
    """ 動画処理を非同期で実行するCeleryタスク """
    in_path  = UPLOAD_DIR / f"{uid}.{in_ext}"
    out_raw  = OUTPUT_DIR / f"{uid}.mp4"
    out_fs   = OUTPUT_DIR / f"{uid}_fs.mp4"

    try:
        # 1. 字幕焼き付け処理 (main2.pyのロジック)
        process_video(
            input_video=in_path,
            output_video=out_raw,
            k_speakers=k_speakers
        )
        
        # 2. Web配信用に再パッケージ (app.pyから移動)
        ffmpeg_cmd = [
            "ffmpeg","-y",
            "-i", str(out_raw),
            "-i", str(in_path),
            "-map","0:v:0","-map","1:a:0?",
            "-c:v","libx264","-pix_fmt","yuv420p","-profile:v","baseline","-level","3.0",
            "-c:a","aac","-b:a","128k",
            "-movflags","+faststart",
            str(out_fs),
        ]
        proc = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        print(f"ffmpeg repackage stdout: {proc.stdout}")

        # ★ 全ての処理が完了した合図として、.doneファイルを作成
        (OUTPUT_DIR / f"{uid}.done").touch()

        return {'status': 'Success', 'file_id': uid}
    
    except subprocess.CalledProcessError as e:
        print(f"Task failed during ffmpeg repackage for {uid}: {e.stderr}")
        if out_raw.exists() and not out_fs.exists():
            out_raw.rename(out_fs)
        return {'status': 'Error', 'message': e.stderr}
    except Exception as e:
        print(f"Task failed for {uid}: {e}")
        return {'status': 'Error', 'message': str(e)}