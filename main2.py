# -*- coding: utf-8 -*-
"""
main2.py (worker)
- CLI usage:
    python main2.py --in <input_video_path> --out <output_video_path> [--device cpu|cuda] [--speakers K] [--font FONT_PATH]

This script transcribes the audio, estimates speakers, builds an ASS subtitle file with colors per speaker,
then burns the subtitles into the input video via ffmpeg to produce the output file.
"""
import os
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List

# Third-party deps (must be installed in the environment):
#   faster_whisper, speechbrain, torchaudio, scikit-learn, pysubs2, pydub, pillow, numpy, scipy
import numpy as np
from PIL import ImageFont
import pysubs2
from faster_whisper import WhisperModel
from speechbrain.pretrained import EncoderClassifier
# 変更後
# from speechbrain.inference.classifiers import EncoderClassifier
import torchaudio
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster, linkage
from pydub import AudioSegment

def ffprobe_video_size(video_path: Path):
    """Return (width, height) using ffprobe."""
    ffprobe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(video_path)
    ]
    res = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
    res.check_returncode()
    info = json.loads(res.stdout)
    w = info["streams"][0]["width"]
    h = info["streams"][0]["height"]
    return int(w), int(h)

def safe_image_font(font_path: Optional[str], size: int) -> ImageFont.ImageFont:
    """
    Try to load the specified TTF/OTF font. If unavailable, try common defaults.
    Fallback to load_default().
    """
    tried = []
    def _try(p):
        if not p:
            return None
        p = str(p)
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            tried.append(p)
            return None

    # 1) requested
    font = _try(font_path)
    if font: return font

    # 2) Common Japanese-capable fonts
    candidates = [
        # Windows
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
        # Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        # macOS
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴ ProN W3.otf",
    ]
    for c in candidates:
        font = _try(c)
        if font: return font

    # 3) fallback
    return ImageFont.load_default()

def wrap_text_for_ass(text: str, font: ImageFont.ImageFont, max_pixel_width: int) -> str:
    """
    Greedy line wrap by pixel width, return ASS string with '\\N' newlines.
    """
    lines: List[str] = []
    cur = ""
    for ch in text:
        test = cur + ch
        try:
            bbox = font.getbbox(test)
            width = bbox[2] - bbox[0]
        except Exception:
            # If font can't measure, fallback to char-count threshold (roughly 2/3 width per char)
            width = len(test) * 12
        if width > max_pixel_width and cur:
            lines.append(cur)
            cur = ch
        else:
            cur = test
    if cur:
        lines.append(cur)
    return "\\N".join(lines)

def get_embeddings_fine(wav_file: Path, start: float, end: float, segment_length: float = 0.5, classifier=None):
    """
    Average ECAPA embeddings across small chunks inside [start, end].
    """
    if classifier is None:
        raise ValueError("EncoderClassifier is required")
    signal, sr = torchaudio.load(str(wav_file))
    embs = []
    t = start
    while t < end:
        s = int(t * sr)
        e = int(min(t + segment_length, end) * sr)
        seg = signal[:, s:e]
        if seg.shape[1] > 0:
            emb = classifier.encode_batch(seg)
            embs.append(emb.squeeze().detach().cpu().numpy())
        t += segment_length

    if not embs:
        return None
    embs = np.stack(embs, axis=0)
    embs = np.mean(embs, axis=0)
    # L2 normalize
    n = np.linalg.norm(embs) + 1e-12
    return embs / n

def cluster_speakers(embeddings: np.ndarray, k: int) -> np.ndarray:
    """
    Try KMeans and hierarchical (Ward), pick the one with better silhouette.
    """
    X = normalize(embeddings.reshape(embeddings.shape[0], -1), axis=1)
    kmeans = KMeans(n_clusters=k, n_init=50, random_state=42).fit(X)
    labels_km = kmeans.labels_

    Z = linkage(X, method="ward")
    labels_hier = fcluster(Z, t=k, criterion="maxclust") - 1

    def safe_silhouette(X, labels):
        # Avoid silhouette if only one cluster
        if len(set(labels)) < 2:
            return -1.0
        return silhouette_score(X, labels)

    sk = safe_silhouette(X, labels_km)
    sh = safe_silhouette(X, labels_hier)
    return labels_hier if sh > sk else labels_km

def burn_subtitles(input_video: Path, ass_path: Path, output_video: Path):
    """
    Use ffmpeg to burn subtitles; keep audio from input if possible.
    """
    # Escape path for FFmpeg subtitles filter
    ass_escaped = str(ass_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace(",", "\\,")
    vf_expr = f"subtitles='{ass_escaped}'"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-vf", vf_expr,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "baseline", "-level", "3.0",
        "-c:a", "copy", # 音声は再エンコードせずコピーする
        "-movflags", "+faststart",
        str(output_video),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

def process_video(input_video: Path, output_video: Path, device: str = "cpu", k_speakers: int = 2, font_path: Optional[str] = None):
    """
    Full pipeline.
    """
    input_video = input_video.resolve()
    output_video = output_video.resolve()

    with tempfile.TemporaryDirectory(prefix="m2_") as td:
        tmpdir = Path(td)
        wav_path = tmpdir / "audio.wav"

        # 1) Extract audio via pydub (requires ffmpeg installed)
        print(f"Extracting audio from {input_video} to {wav_path}")
        audio = AudioSegment.from_file(str(input_video))
        audio.export(str(wav_path), format="wav")

        # 2) Transcribe
        print("Transcribing audio...")
        model = WhisperModel("small", device=device)
        segments, _ = model.transcribe(str(wav_path), beam_size=5, language="ja")
        transcripts = []
        for i, seg in enumerate(segments):
            # faster-whisper returns float seconds
            transcripts.append({
                "id": int(i),
                "start": float(seg.start),
                "end": float(seg.end),
                "text": (seg.text or "").strip(),
            })

        # filter out empty
        transcripts = [t for t in transcripts if t["end"] > t["start"] and t["text"]]
        if not transcripts:
            print("No text transcribed. Aborting subtitle generation.")
            # 空の動画をコピーして終了
            subprocess.run(["cp", str(input_video), str(output_video)], check=True)
            return

        # 3) Speaker embeddings (ECAPA)
        print("Generating speaker embeddings...")
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(tmpdir / "pretrained_models/spkrec-ecapa-voxceleb")
        )
        embs = []
        valid = []
        for t in transcripts:
            e = get_embeddings_fine(wav_path, t["start"], t["end"], segment_length=0.5, classifier=classifier)
            if e is not None:
                embs.append(e)
                valid.append(t)

        if not valid:
            # No valid embeddings; create single-speaker captions
            print("No valid embeddings found. Treating as single speaker.")
            valid = transcripts
            speaker_labels = np.zeros(len(valid), dtype=int)
            k_speakers = 1
        else:
            print(f"Clustering {len(embs)} segments into {k_speakers} speakers...")
            embs = np.array(embs)
            speaker_labels = cluster_speakers(embs, min(len(embs), int(k_speakers)))

        # 4) Speaker colors (ASS BGR format like &HBBGGRR&)
        palette = [
            "{\\c&H00FF00&}",  # Green
            "{\\c&HFF0000&}",  # Blue
            "{\\c&H00FFFF&}",  # Yellow
            "{\\c&HFF00FF&}",  # Magenta
            "{\\c&HFFFF00&}",  # Cyan
            "{\\c&H0000FF&}",  # Red
            "{\\c&HFFFFFF&}",  # White
            "{\\c&H808080&}",  # Gray
        ]

        # 5) Video dimensions
        vw, vh = ffprobe_video_size(input_video)

        # 6) Font + wrapping
        font_size = max(24, int(vh / 25))
        font = safe_image_font(font_path, size=font_size)

        # 7) Build ASS
        print("Building ASS subtitle file...")
        subs = pysubs2.SSAFile()
        subs.info["PlayResX"] = vw
        subs.info["PlayResY"] = vh

        default_style = pysubs2.SSAStyle(
            fontname="Arial", # より一般的なフォントに変更
            fontsize=font_size,
            primarycolor=pysubs2.Color(255, 255, 255),
            outlinecolor=pysubs2.Color(0, 0, 0),
            backcolor=pysubs2.Color(0, 0, 0, 128),
            bold=True,
            outline=2,
            shadow=1,
            alignment=2,  # bottom-center
            marginv=max(20, int(vh * 0.05)),
        )
        subs.styles["Default"] = default_style

        for i, t in enumerate(valid):
            start_ms = int(t["start"] * 1000)
            end_ms = int(t["end"] * 1000)
            speaker_idx = int(speaker_labels[i])
            color_tag = palette[speaker_idx % len(palette)]
            wrapped = wrap_text_for_ass(t["text"], font, int(vw * 0.9))
            subs.append(pysubs2.SSAEvent(start=start_ms, end=end_ms, text=f"{color_tag}{wrapped}", style="Default"))

        ass_path = tmpdir / "subs.ass"
        subs.save(str(ass_path))

        # 8) Burn subtitles
        print(f"Burning subtitles into {output_video}...")
        burn_subtitles(input_video, ass_path, output_video)
        print("Subtitle burning complete.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input video path")
    ap.add_argument("--out", dest="out", required=True, help="Output video path")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device for faster-whisper")
    ap.add_argument("--speakers", type=int, default=2, help="Expected number of speakers (clusters)")
    ap.add_argument("--font", type=str, default=None, help="TTF/OTF font path for width measurement / rendering")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    if not inp.exists():
        print(f"[main2] Input video not found: {inp}", file=sys.stderr)
        sys.exit(2)

    try:
        process_video(inp, out, device=args.device, k_speakers=args.speakers, font_path=args.font)
        print(f"[main2] OK -> {out}")
    except subprocess.CalledProcessError as e:
        print(f"[main2] ffmpeg error: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[main2] error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()