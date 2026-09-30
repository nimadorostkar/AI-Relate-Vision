"""Real-time playback and offline export.

Live playback (video files)
    Three threads. A reader decodes frames ahead of the screen, an inference
    worker always takes the NEWEST decoded frame (so it never falls behind, it
    just samples more sparsely on a slow machine), and the main thread shows
    the video at its native frame rate with a short fixed delay. The delay
    means every inference result is applied exactly on the frame it was
    computed from, so boxes line up with the picture; between results the
    Kalman tracker carries boxes and relations forward. On a 4-core laptop
    CPU the model runs ~3-5 times a second while the video plays at full fps.

Webcam
    Same worker; no delay (you would notice it), results land when ready.

Export
    Deterministic: inference every --stride frames, every frame rendered,
    written as H.264 MP4 plus a JSON log of relation events and a summary.
    Slower than real time, but the output is complete and smooth.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from relatevision.engine import PRESETS, REPO, RelationEngine
from relatevision.overlay import Overlay, OverlayConfig
from relatevision.video import VideoInfo, VideoWriter, iter_frames, probe

WINDOW = "AI-Relate-Vision  (q quit | space pause | h help)"
HELP = [
    "q / ESC  quit (n: next video)     space  pause",
    "[ / ]    relation threshold -/+   + / -  more / fewer relations",
    "p        predicate preset         g      spatial relations on/off",
    "a        all object boxes         s      save screenshot",
    "h        this help / HUD",
    "Based on RelateAnything by Maelic Neau - github.com/Maelic/RelateAnything",
]
PRESET_ORDER = list(PRESETS)


def _print_event(ev: dict) -> None:
    if ev["event"] == "start":
        print(f"  [{ev['t']:7.2f}s] {ev['subject']}#{ev['subject_id']} "
              f"--{ev['predicate']}--> {ev['object']}#{ev['object_id']} "
              f"({ev['score']:.2f})", flush=True)


class _Worker(threading.Thread):
    """Runs the engine on the most recent frame handed to it."""

    def __init__(self, engine: RelationEngine):
        super().__init__(daemon=True)
        self.engine = engine
        self._cv = threading.Condition()
        self._job = None                     # (idx, frame)
        self.results: deque = deque()        # (idx, FrameResult)
        self.busy = False
        self.stop_flag = False
        self.lock = threading.Lock()         # held while running the engine
        self.ms = deque(maxlen=20)

    def submit(self, idx: int, frame: np.ndarray) -> None:
        with self._cv:
            self._job = (idx, frame)
            self._cv.notify()

    def idle(self) -> bool:
        with self._cv:
            return self._job is None and not self.busy

    def run(self) -> None:
        while True:
            with self._cv:
                while self._job is None and not self.stop_flag:
                    self._cv.wait(0.1)
                if self.stop_flag:
                    return
                idx, frame = self._job
                self._job = None
                self.busy = True
            with self.lock:
                t = time.perf_counter()
                res = self.engine(frame)
                self.ms.append((time.perf_counter() - t) * 1e3)
            self.results.append((idx, res))
            with self._cv:
                self.busy = False

    def rate(self) -> float:
        return 1000.0 / (sum(self.ms) / len(self.ms)) if self.ms else 0.0

    def stop(self) -> None:
        with self._cv:
            self.stop_flag = True
            self._cv.notify()


class _Reader(threading.Thread):
    """Decodes ahead into a bounded buffer."""

    def __init__(self, path: Path, info: VideoInfo, ahead: int):
        super().__init__(daemon=True)
        self.path, self.info, self.ahead = path, info, max(ahead, 8)
        self.frames: Dict[int, np.ndarray] = {}
        self.newest = -1
        self.done = False
        self.consumed = 0                    # display position
        self.stop_flag = False
        self.cv = threading.Condition()

    def run(self) -> None:
        try:
            for i, fr in enumerate(iter_frames(self.path, self.info)):
                with self.cv:
                    while (i - self.consumed > self.ahead) and not self.stop_flag:
                        self.cv.wait(0.05)
                    if self.stop_flag:
                        return
                    self.frames[i] = fr
                    self.newest = i
        finally:
            with self.cv:
                self.done = True

    def advance(self, pos: int) -> None:
        with self.cv:
            self.consumed = pos
            for k in [k for k in self.frames if k < pos - 1]:
                del self.frames[k]
            self.cv.notify_all()


class Controls:
    """Keyboard handling shared by file and webcam playback."""

    def __init__(self, engine: RelationEngine, ov: Overlay, worker: _Worker,
                 preset: str):
        self.engine, self.ov, self.worker = engine, ov, worker
        self.preset = preset if preset in PRESET_ORDER else "custom"
        self.show_help = False
        self.show_hud = True
        self.paused = False
        self._spatial = ov.cfg.max_spatial or 3

    def handle(self, key: int, frame_out: Optional[np.ndarray]) -> str:
        if key in (27, ord("q")):
            return "quit"
        if key == ord("n"):
            return "next"
        if key == ord(" "):
            self.paused = not self.paused
        elif key == ord("["):
            self.ov.set_threshold(self.ov.cfg.on_thr - 0.03)
        elif key == ord("]"):
            self.ov.set_threshold(self.ov.cfg.on_thr + 0.03)
        elif key in (ord("+"), ord("=")):
            self.ov.set_max_edges(self.ov.cfg.max_edges + 1)
        elif key == ord("-"):
            self.ov.set_max_edges(self.ov.cfg.max_edges - 1)
        elif key == ord("g"):
            if self.ov.cfg.max_spatial:
                self._spatial = self.ov.cfg.max_spatial
                self.ov.set_max_spatial(0)
            else:
                self.ov.set_max_spatial(self._spatial)
        elif key == ord("a"):
            self.ov.cfg.all_boxes = not self.ov.cfg.all_boxes
        elif key == ord("h"):
            self.show_help = not self.show_help
        elif key == ord("p"):
            i = PRESET_ORDER.index(self.preset) if self.preset in PRESET_ORDER else -1
            self.preset = PRESET_ORDER[(i + 1) % len(PRESET_ORDER)]
            with self.worker.lock:           # never swap under a running inference
                self.engine.set_predicates(PRESETS[self.preset])
                self.worker.results.clear()
            self.ov.vocabulary_changed()
            print(f"[player] preset -> {self.preset} "
                  f"({len(self.engine.predicates)} predicates)")
        elif key == ord("s") and frame_out is not None:
            out = REPO / "outputs" / f"screenshot_{int(time.time())}.jpg"
            out.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(out), frame_out)
            print(f"[player] saved {out}")
        return ""

    def hud(self, extra: str = "") -> List[str]:
        if self.show_help:
            return HELP
        if not self.show_hud:
            return []
        eng = self.engine
        r = self.ov.last
        lines = [f"relations/s {self.worker.rate():4.1f}   "
                 + (f"det {r.det_ms:4.0f} ms  rel {r.rel_ms:4.0f} ms" if r else "warming up"),
                 f"preset {self.preset} ({len(eng.predicates)})   thr {self.ov.cfg.on_thr:.2f}"
                 f"   edges<={self.ov.cfg.max_edges}   spatial {'on' if self.ov.cfg.max_spatial else 'off'}"]
        if extra:
            lines.append(extra)
        return lines


def play_file(engine: RelationEngine, path: Path, ov_cfg: OverlayConfig,
              max_width: int, delay: float, preset: str) -> str:
    info = probe(path, max_width)
    fps = info.fps
    print(f"[player] {path.name}: {info.src_width}x{info.src_height} -> "
          f"{info.width}x{info.height} @ {fps:.2f} fps"
          + (f", {info.duration:.1f}s" if info.duration else ""))
    ov = Overlay(engine, info.width, info.height, ov_cfg)
    worker = _Worker(engine); worker.start()
    delay_frames = max(1, int(round(delay * fps)))
    reader = _Reader(path, info, ahead=delay_frames + int(fps))
    reader.start()
    ctl = Controls(engine, ov, worker, preset)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, info.width, info.height)
    # Let the reader get ahead, and give frame 0 a result before the clock
    # starts, so the overlay is there from the first frame.
    while reader.newest < delay_frames and not reader.done:
        time.sleep(0.01)
    if 0 in reader.frames:
        worker.submit(0, reader.frames[0])
        while not worker.results and worker.is_alive():
            time.sleep(0.005)
    t0 = time.perf_counter()
    pos = 0                    # next frame to display
    last_sub = 0
    n_shown = n_obs0 = 0
    n_obs0 = ov.n_obs
    out = None
    last_frame = None
    action = ""
    shown = deque(maxlen=30)
    try:
        while True:
            if ctl.paused:
                p0 = time.perf_counter()
                if last_frame is not None:
                    out = ov.render(last_frame, ctl.hud("PAUSED"))
                    cv2.imshow(WINDOW, out)
                action = ctl.handle(cv2.waitKey(30) & 0xFF, out)
                t0 += time.perf_counter() - p0
                if action:
                    break
                continue

            # 1. keep the worker fed with the newest decoded frame
            newest = reader.newest
            if newest > last_sub and newest >= pos and worker.idle():
                fr = reader.frames.get(newest)
                if fr is not None:
                    worker.submit(newest, fr)
                    last_sub = newest

            # 2. which frame belongs on screen now
            target = int((time.perf_counter() - t0) * fps)
            if reader.done and pos > reader.newest:
                break
            target = min(target, reader.newest)
            if target < pos:
                key = cv2.waitKey(max(1, int((pos - target) * 1000 / fps) - 1)) & 0xFF
                action = ctl.handle(key, out)
                if action:
                    break
                continue

            # 3. apply results and advance the filters through every frame up
            #    to `target` (frames we are too slow to draw are skipped, but
            #    the tracker still steps through them)
            frame = None
            for i in range(pos, target + 1):
                while worker.results and worker.results[0][0] <= i:
                    _, res = worker.results.popleft()
                    ov.observe(res)
                ov.step(i / fps)
                for ev in ov.events:
                    _print_event(ev)
                ov.events.clear()
            frame = reader.frames.get(target)
            pos = target + 1
            reader.advance(pos)
            if frame is None:
                continue
            last_frame = frame
            n_shown += 1
            shown.append(time.perf_counter())
            disp_fps = (len(shown) - 1) / max(shown[-1] - shown[0], 1e-6) if len(shown) > 1 else 0
            out = ov.render(frame, ctl.hud(f"video {target / fps:6.1f}s   display {disp_fps:4.1f} fps"))
            cv2.imshow(WINDOW, out)
            key = cv2.waitKey(1) & 0xFF
            action = ctl.handle(key, out)
            if action:
                break
    finally:
        reader.stop_flag = True
        reader.advance(10 ** 9)
        worker.stop()
    el = max(time.perf_counter() - t0, 1e-6)
    print(f"[player] {pos} frames in {el:.1f}s: displayed {n_shown} "
          f"({n_shown / el:.1f} fps), {ov.n_obs - n_obs0} model updates "
          f"({(ov.n_obs - n_obs0) / el:.1f}/s)")
    return action


def play_webcam(engine: RelationEngine, cam: int, ov_cfg: OverlayConfig,
                max_width: int, preset: str) -> None:
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {cam} (macOS: allow camera "
                         "access for your terminal in System Settings > Privacy)")
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("camera returned no frames")
    h, w = frame.shape[:2]
    s = min(1.0, max_width / max(w, h))
    W, H = int(w * s) // 2 * 2, int(h * s) // 2 * 2
    ov = Overlay(engine, W, H, ov_cfg)
    worker = _Worker(engine); worker.start()
    ctl = Controls(engine, ov, worker, preset)
    t0 = time.perf_counter()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    out = None
    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (w, h) != (W, H):
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            if worker.idle():
                worker.submit(i, frame)
            while worker.results:
                _, res = worker.results.popleft()
                ov.observe(res)
            ov.step(time.perf_counter() - t0)
            for ev in ov.events:
                _print_event(ev)
            ov.events.clear()
            if not ctl.paused:
                out = ov.render(frame, ctl.hud())
            cv2.imshow(WINDOW, out)
            if ctl.handle(cv2.waitKey(1) & 0xFF, out) in ("quit", "next"):
                break
            i += 1
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()


def export(engine: RelationEngine, path: Path, out_dir: Path, ov_cfg: OverlayConfig,
           max_width: int, stride: int, preview: bool, hud: bool) -> Path:
    info = probe(path, max_width)
    fps = info.fps
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    out_mp4 = out_dir / f"{stem}_relations.mp4"
    out_log = out_dir / f"{stem}_relations.jsonl"
    out_sum = out_dir / f"{stem}_summary.json"
    n = info.frames
    print(f"[export] {path.name}: {info.width}x{info.height} @ {fps:.2f} fps, "
          f"{n or '?'} frames, inference every {stride} frame(s)")
    ov = Overlay(engine, info.width, info.height, ov_cfg)
    writer = VideoWriter(out_mp4, info.width, info.height, fps)
    t_start = time.perf_counter()
    inf_ms: List[float] = []
    i = 0
    with open(out_log, "w") as log:
        for i, frame in enumerate(iter_frames(path, info)):
            t = i / fps
            if i % stride == 0:
                res = engine(frame)
                inf_ms.append(res.total_ms)
                ov.observe(res)
            ov.step(t)
            for ev in ov.events:
                log.write(json.dumps({"type": "relation_" + ev.pop("event"),
                                      "frame": i, **ev}) + "\n")
            ov.events.clear()
            if i % stride == 0:
                log.write(json.dumps({"type": "frame", "frame": i, "t": round(t, 3),
                                      "objects": ov.objects(),
                                      "relations": ov.current()}) + "\n")
            hud_lines = None
            if hud:
                r = ov.last
                hud_lines = [f"t {t:6.1f}s   det {r.det_ms:4.0f} ms  rel {r.rel_ms:4.0f} ms"
                             if r else f"t {t:6.1f}s"]
            frame_out = ov.render(frame, hud_lines)
            writer.write(frame_out)
            if preview:
                cv2.imshow(WINDOW, frame_out)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    preview = False
                    cv2.destroyAllWindows()
            if i % 30 == 0 and i:
                el = time.perf_counter() - t_start
                speed = (i + 1) / el
                eta = f", ETA {max(0, n - i - 1) / speed:5.0f}s" if n else ""
                print(f"  frame {i:6d}/{n or '?'}  {speed:5.1f} fps{eta}   "
                      f"relations now: {len(ov.current())}", flush=True)
    writer.close()
    el = time.perf_counter() - t_start
    summary = {
        "video": str(path), "frames": i + 1, "fps": fps,
        "size": [info.width, info.height], "stride": stride,
        "processing_seconds": round(el, 1),
        "mean_inference_ms": round(float(np.mean(inf_ms)), 1) if inf_ms else None,
        "predicates": engine.predicates,
        "top_relations": ov.summary(fps),
    }
    out_sum.write_text(json.dumps(summary, indent=1))
    if preview:
        cv2.destroyAllWindows()
    print(f"[export] {i + 1} frames in {el:.1f}s ({(i + 1) / el:.1f} fps)")
    print(f"[export] video   -> {out_mp4}")
    print(f"[export] events  -> {out_log}")
    print(f"[export] summary -> {out_sum}")
    for r in summary["top_relations"][:10]:
        print(f"   {r['seconds_visible']:6.1f}s  {r['subject']} --{r['predicate']}--> "
              f"{r['object']}  (best {r['best_score']:.2f})")
    return out_mp4
