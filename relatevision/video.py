"""Video in/out. OpenCV first (fast, handles phone MP4/MOV incl. rotation);
ffmpeg (bundled by imageio-ffmpeg) as the fallback for codecs the OpenCV wheel
cannot decode, such as AV1 or some WebM files."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpg",
              ".mpeg", ".wmv", ".flv", ".3gp", ".ts", ".mts"}


def collect_inputs(paths: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.iterdir()
                          if f.suffix.lower() in VIDEO_EXTS and not f.name.startswith("."))
        elif p.exists():
            out.append(p)
        else:
            print(f"[video] not found, skipped: {p}")
    return out


@dataclass
class VideoInfo:
    width: int          # output (possibly downscaled) size
    height: int
    fps: float
    frames: int         # 0 when unknown
    src_width: int
    src_height: int

    @property
    def duration(self) -> float:
        return self.frames / self.fps if self.frames and self.fps else 0.0


def _even(n: float) -> int:
    return max(2, int(n) // 2 * 2)


def _fit(w: int, h: int, max_width: int) -> tuple[int, int]:
    s = min(1.0, max_width / max(w, h)) if max_width else 1.0
    return _even(w * s), _even(h * s)


def probe(path: Path, max_width: int = 960) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    if ok:
        h, w = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
    else:
        cap.release()
        import imageio_ffmpeg
        gen = imageio_ffmpeg.read_frames(str(path), pix_fmt="bgr24")
        try:
            meta = next(gen)
        finally:
            gen.close()
        w, h = meta["size"]
        fps = float(meta.get("fps") or 30.0)
        n = int(round(float(meta.get("duration") or 0) * fps))
    if not (1.0 <= fps <= 240.0):
        fps = 30.0
    W, H = _fit(w, h, max_width)
    return VideoInfo(W, H, float(fps), max(n, 0), w, h)


def iter_frames(path: Path, info: VideoInfo) -> Iterator[np.ndarray]:
    """Sequential BGR frames at info.width x info.height."""
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    if ok:
        try:
            while ok:
                if frame.shape[1] != info.width or frame.shape[0] != info.height:
                    frame = cv2.resize(frame, (info.width, info.height),
                                       interpolation=cv2.INTER_AREA)
                yield frame
                ok, frame = cap.read()
        finally:
            cap.release()
        return
    cap.release()
    import imageio_ffmpeg
    gen = imageio_ffmpeg.read_frames(
        str(path), pix_fmt="bgr24",
        output_params=["-vf", f"scale={info.width}:{info.height}"])
    next(gen)
    try:
        for buf in gen:
            yield np.frombuffer(buf, np.uint8).reshape(info.height, info.width, 3)
    finally:
        gen.close()


class VideoWriter:
    """H.264 MP4 via ffmpeg (plays everywhere, incl. QuickTime and browsers)."""

    def __init__(self, path: Path, width: int, height: int, fps: float,
                 crf: int = 21):
        import imageio_ffmpeg
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-r", f"{fps:.4f}", "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.path = path

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()
