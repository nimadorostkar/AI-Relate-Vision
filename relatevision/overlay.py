"""Temporal smoothing + drawing, on top of the upstream reel renderer.

The relation head is per-frame and has no memory. deploy/render_video.py
already solves that for offline reels: a Kalman filter per object track
(ByteTrack-style association) and a Kalman "belief" per relation on calibrated
log-odds, with hysteresis so edges do not blink. This module drives those same
classes from a live/async source, and adds an event log (relation appeared /
ended) plus a per-video summary.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from relatevision.engine import FrameResult, RelationEngine  # noqa: F401  (sys.path)
from deploy import render_video as rv  # noqa: E402

# macOS fonts first; the upstream list is Linux paths.
rv.FONTS[:0] = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/Library/Fonts/Arial Bold.ttf"]


@dataclass
class OverlayConfig:
    on_thr: float = 0.45          # a relation must reach this (calibrated) to appear
    off_gap: float = 0.10         # ...and fall this far below it to disappear
    spatial_on_thr: float = 0.42
    max_edges: int = 8
    max_spatial: int = 3          # spatial edges ("on", "behind") get their own budget
    track_high: float = 0.18      # detector confidence needed to start a new track
    max_age: int = 8              # observations a track survives unmatched
    all_boxes: bool = False       # also draw tracked objects that carry no relation
    belief: bool = True           # Kalman belief filter on relations (vs EMA)


class Overlay:
    def __init__(self, engine: RelationEngine, width: int, height: int,
                 cfg: OverlayConfig = OverlayConfig()):
        self.engine, self.W, self.H, self.cfg = engine, width, height, cfg
        self.is_sp = engine.spatial_map()
        self.tracker = rv.Tracker(iou_thr=0.3, merge_iou=0.75,
                                  max_age=cfg.max_age, min_hits=2,
                                  high_conf=cfg.track_high, q=1.0, r=10.0)
        self.book = self._new_book()
        s = max(height, 360)
        self.fonts = (rv._font(max(14, int(s * 0.026))),
                      rv._font(max(13, int(s * 0.025))),
                      rv._font(max(11, int(s * 0.020))))
        self._on: Dict[tuple, dict] = {}         # edges currently shown
        self.events: List[dict] = []             # appeared / ended log
        self.stats: Dict[tuple, dict] = defaultdict(
            lambda: {"frames": 0, "first": None, "last": None, "best": 0.0})
        self.last: Optional[FrameResult] = None
        self.n_obs = 0

    def _new_book(self) -> "rv.EdgeBook":
        c = self.cfg
        return rv.EdgeBook(score_ema=0.35, on_thr=c.on_thr,
                           off_thr=c.on_thr - c.off_gap, on_frames=2,
                           off_frames=6, max_edges=c.max_edges,
                           max_spatial=c.max_spatial,
                           spatial_on_thr=min(c.spatial_on_thr, c.on_thr),
                           belief=c.belief, kf_q=0.05, kf_r=1.0)

    # -- live knobs --------------------------------------------------------
    def set_threshold(self, on_thr: float) -> None:
        self.cfg.on_thr = float(np.clip(on_thr, 0.05, 0.95))
        self.book.on_thr = self.cfg.on_thr
        self.book.off_thr = self.cfg.on_thr - self.cfg.off_gap
        self.book.spatial_on_thr = min(self.cfg.spatial_on_thr, self.cfg.on_thr)

    def set_max_edges(self, n: int) -> None:
        self.cfg.max_edges = self.book.max_edges = max(1, int(n))

    def set_max_spatial(self, n: int) -> None:
        self.cfg.max_spatial = self.book.max_spatial = max(0, int(n))

    def vocabulary_changed(self) -> None:
        """Predicates were swapped: relation beliefs refer to column indices
        of the old vocabulary, so start the relation book over (tracks stay)."""
        self.is_sp = self.engine.spatial_map()
        self.book = self._new_book()

    # -- per inference / per frame ----------------------------------------
    def observe(self, res: FrameResult) -> None:
        det2track = self.tracker.observe(
            res.boxes, res.labels, res.confs,
            frame_area=float(self.W * self.H), max_frac=0.85)
        if res.raw is not None or not self.cfg.belief:
            # Under 2 boxes nothing is measurable: with the belief filter on,
            # hold every relation rather than read it as evidence against.
            self.book.observe(res.triplets, det2track, self.is_sp,
                              res.raw if self.cfg.belief else None)
        self.last = res
        self.n_obs += 1

    def step(self, t: float) -> None:
        """Advance one displayed frame at video time t (seconds)."""
        self.tracker.step(self.W, self.H)
        self.book.step({tr.tid for tr in self.tracker.tracks})
        tracks = self.tracker.by_id()
        now = {}
        for e in self.book.visible():
            if not e.on:
                continue
            ts, to = tracks.get(e.sub), tracks.get(e.obj)
            if ts is None or to is None:
                continue
            key = (e.sub, e.pred, e.obj)
            now[key] = {"subject": ts.label, "subject_id": e.sub,
                        "predicate": e.pred, "object": to.label,
                        "object_id": e.obj, "score": round(max(e.shown, e.score), 3),
                        "spatial": e.spatial}
        counted = set()
        for key, d in now.items():
            if key not in self._on:
                self.events.append({"t": round(t, 2), "event": "start", **d})
            lk = (d["subject"], d["predicate"], d["object"])
            st = self.stats[lk]
            if lk not in counted:          # two person/helmet pairs = one frame
                counted.add(lk)
                st["frames"] += 1
            st["first"] = t if st["first"] is None else st["first"]
            st["last"] = t
            st["best"] = max(st["best"], d["score"])
        for key, d in self._on.items():
            if key not in now:
                self.events.append({"t": round(t, 2), "event": "end", **d})
        self._on = now

    def current(self) -> List[dict]:
        return sorted(self._on.values(), key=lambda d: -d["score"])

    def objects(self) -> List[dict]:
        return [{"id": t.tid, "label": t.label, "conf": round(t.conf, 3),
                 "box": [round(float(v), 1) for v in t.box]}
                for t in self.tracker.tracks
                if t.hits >= self.tracker.min_hits and t.misses <= 2]

    # -- drawing -------------------------------------------------------------
    def render(self, frame: np.ndarray, hud: Optional[List[str]] = None) -> np.ndarray:
        if self.cfg.all_boxes:
            frame = frame.copy()
            carrying = {e.sub for e in self.book.visible()} | \
                       {e.obj for e in self.book.visible()}
            for t in self.tracker.tracks:
                if t.alpha <= 0.3 or t.tid in carrying:
                    continue
                x0, y0, x1, y1 = [int(v) for v in t.box]
                cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(frame, t.label, (x0 + 3, max(12, y0 + 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        out = rv.render_frame(frame, self.tracker, self.book, self.fonts)
        if hud:
            draw_hud(out, hud)
        return out

    def summary(self, fps: float, top: int = 40) -> List[dict]:
        rows = []
        for (s, p, o), st in self.stats.items():
            rows.append({"subject": s, "predicate": p, "object": o,
                         "seconds_visible": round(st["frames"] / fps, 2),
                         "first_seen": round(st["first"], 2),
                         "last_seen": round(st["last"], 2),
                         "best_score": round(st["best"], 3)})
        rows.sort(key=lambda r: -r["seconds_visible"])
        return rows[:top]


def draw_hud(img: np.ndarray, lines: List[str]) -> None:
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    lh = 18
    w = max(cv2.getTextSize(l, font, scale, th)[0][0] for l in lines) + 16
    h = lh * len(lines) + 10
    roi = img[0:h, 0:min(w, img.shape[1])]
    roi[:] = (roi * 0.35).astype(np.uint8)
    for i, l in enumerate(lines):
        cv2.putText(img, l, (8, 17 + i * lh), font, scale, (120, 255, 170), th,
                    cv2.LINE_AA)
