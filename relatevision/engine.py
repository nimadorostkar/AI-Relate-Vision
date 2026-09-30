"""The fast CPU pipeline: detector -> boxes -> open-vocabulary relations.

Built for a laptop without an NVIDIA GPU. Two choices do most of the work:

* The relation head runs on OpenVINO with its image input reshaped from the
  exported 448 px down to 320 px. The DINOv3 backbone is ~80% of the head's
  cost and scales with the token count, so 320 px is ~1.8x faster; on the
  sample images the top-10 relations agree 8-10/10 with 448 px.
* The YOLO-World detector (the 497 RA-4M object classes) runs on onnxruntime
  at 416 px instead of 640 px, ~2x faster at the same box count on typical
  footage.

Everything else - letterbox, NMS, the score contract, per-predicate
calibrated thresholds, decoding - is the upstream code in deploy/, reused as is.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from deploy.postprocess import ThresholdConfig, Triplet, decode  # noqa: E402
from deploy.runtime import (DetectorConfig, OnnxDetector,  # noqa: E402
                            OnnxRelationHead)

DIST = REPO / "deploy" / "dist" / "relsgg-vits16plus"
DET_DIR = REPO / "deploy" / "dist" / "detector-local"
OV_CACHE = REPO / "checkpoints" / "ov_cache"

# quality preset -> (detector px, relation px)
QUALITY = {
    "fast": (416, 256),
    "balanced": (416, 320),
    "best": (640, 448),
}

# Predicate presets. Every entry is intersected with the shipped bank (243
# phrases), so a phrase the bank lacks is dropped rather than fatal.
INTERACTION = [
    "riding", "holding", "carrying", "playing", "using", "wearing",
    "sitting on", "sitting at", "sitting in", "standing on", "looking at",
    "watching", "eating", "drinking from", "cutting", "pushing", "pulling",
    "climbing", "talking to", "walking with", "leaning against", "reading",
    "playing with", "feeding", "petting", "hitting", "kicking", "driving",
    "lying on", "hugging", "touching", "throwing", "walking on",
    "standing beside", "walking past", "posing with", "smiling at",
    "part of", "resting on", "covering", "inside", "contained in",
    "hanging from", "surrounding", "attached to",
]
SPATIAL = ["on", "on top of", "in front of", "behind", "beside",
           "to the left of", "to the right of", "above", "below", "under"]
PRESETS = {"all": INTERACTION + SPATIAL, "interaction": INTERACTION,
           "spatial": SPATIAL}


def _detector_path(size: int) -> Path:
    p = DET_DIR / (f"detector_{size}.onnx" if size != 640 else "detector.onnx")
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p.relative_to(REPO)} - run ./setup_mac.sh first "
            "(it builds the detector locally)")
    return p


def _ort_session(path: str, threads: int):
    """onnxruntime session whose worker threads sleep between runs. Spinning
    (the default) keeps cores busy after each call, stealing them from the
    OpenVINO head that runs right after and from the thread drawing video."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


class QuietOnnxDetector(OnnxDetector):
    def _build_engine(self, path, threads, providers) -> None:
        self.sess = _ort_session(path, threads)
        self.iname = self.sess.get_inputs()[0].name


class QuietOnnxRelationHead(OnnxRelationHead):
    def _build_engine(self, path, threads, providers) -> List[str]:
        self.sess = _ort_session(path, threads)
        self.input_names = [i.name for i in self.sess.get_inputs()]
        return [o.name for o in self.sess.get_outputs()]


class SizedOVRelationHead(OnnxRelationHead):
    """The released relation graph on OpenVINO, with the image input reshaped.

    Subclasses the ONNX head and overrides only the engine hooks, so the
    preprocessing (make_feed) and the score contract stay the upstream ones;
    make_feed resizes to self.img_size, which is set to the reshaped size.
    """

    def __init__(self, model_path: str, bank_path: str, img_size: int,
                 threads: int = 0):
        self._target = int(img_size)
        super().__init__(model_path, bank_path=bank_path, threads=threads)

    def _build_engine(self, path, threads, providers) -> List[str]:
        import openvino as ov
        core = ov.Core()
        OV_CACHE.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(OV_CACHE)})
        model = core.read_model(path)
        if self._target != self.img_size:
            S = self._target
            model.reshape({"image": [1, 3, S, S]})
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if threads:
            cfg["INFERENCE_NUM_THREADS"] = int(threads)
        self.compiled = core.compile_model(model, "CPU", cfg)
        self.req = self.compiled.create_infer_request()
        self.input_names = [i.get_any_name() for i in self.compiled.inputs]
        self.img_size = self._target
        return [o.get_any_name() for o in self.compiled.outputs]

    def _run_engine(self, feed):
        res = self.req.infer(feed)
        return [np.asarray(res[o]) for o in self.compiled.outputs]


@dataclass
class FrameResult:
    boxes: np.ndarray
    confs: np.ndarray
    labels: List[str]
    triplets: List[Triplet]
    raw: Optional[tuple]
    det_ms: float = 0.0
    rel_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.det_ms + self.rel_ms


@dataclass
class EngineConfig:
    quality: str = "balanced"
    backend: str = "openvino"          # relation head: openvino | onnx
    threads: int = 0                   # 0 = physical cores
    det_conf: float = 0.12             # 497 classes split the confidence mass
    det_iou: float = 0.5
    max_boxes: int = 24
    topk: int = 24
    max_per_pair: int = 2
    floor: float = 0.30                # decode floor for uncalibrated phrases
    predicates: Optional[Sequence[str]] = None
    preset: str = "all"


def physical_cores() -> int:
    try:
        import subprocess
        out = subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                             capture_output=True, text=True, timeout=2)
        n = int(out.stdout.strip())
        if n > 0:
            return n
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


class RelationEngine:
    """Detector + relation head + calibrated decoding. Not thread-safe: call
    it from one thread (the player's inference worker)."""

    def __init__(self, cfg: EngineConfig = EngineConfig()):
        self.cfg = cfg
        if cfg.quality not in QUALITY:
            raise ValueError(f"quality must be one of {list(QUALITY)}")
        det_size, rel_size = QUALITY[cfg.quality]
        threads = cfg.threads or physical_cores()
        self.threads = threads

        rel_onnx = DIST / "relateanything.onnx"
        if not rel_onnx.exists():
            raise FileNotFoundError(
                f"missing {rel_onnx.relative_to(REPO)} - run ./setup_mac.sh first")
        bank = str(DIST / "predicate_bank.npz")

        self.det = QuietOnnxDetector(str(_detector_path(det_size)), threads=threads)
        backend = cfg.backend
        if backend == "openvino":
            try:
                import openvino  # noqa: F401
            except ImportError:
                print("[engine] openvino not installed - using onnxruntime @448px")
                backend = "onnx"
        if backend == "openvino":
            # fp16 IR = compressed weights; the math is fp32 on CPU. (int8 PTQ
            # was measured on an i5-8257U: slower than fp32 - no VNNI - and it
            # changed the decoded graph, so it is not offered.)
            xml = DIST / "relateanything_fp16.xml"
            src = str(xml if xml.exists() else rel_onnx)
            self.rel = SizedOVRelationHead(src, bank, rel_size, threads=threads)
        else:
            self.rel = QuietOnnxRelationHead(str(rel_onnx), bank_path=bank,
                                             threads=threads)
        self.backend = backend
        self.det_cfg = DetectorConfig(conf=cfg.det_conf, iou=cfg.det_iou,
                                      max_det=cfg.max_boxes)
        c = self.rel.contract
        self.thr_cfg = ThresholdConfig(threshold=cfg.floor, topk=cfg.topk,
                                       pair_weight=1.0,
                                       max_per_pair=cfg.max_per_pair,
                                       calib_a=c.calib_a, calib_b=c.calib_b)
        self._bank_thr = self._calibrated_thresholds()
        self.set_predicates(cfg.predicates or PRESETS[cfg.preset])

    # -- vocabulary --------------------------------------------------------
    def _calibrated_thresholds(self) -> Dict[str, float]:
        """Per-predicate best-F1 thresholds from the bank, mapped into the
        calibrated score's units (same affine map as deploy/runtime.py's
        ScenePipeline), for EVERY phrase in the bank - so switching presets
        keeps each phrase's own operating point."""
        bank = self.rel.bank
        if bank is None or bank.get("thr") is None:
            return {}
        a, b = self.thr_cfg.calib_a, self.thr_cfg.calib_b
        out = {}
        for n, v in zip(bank["names"], bank["thr"]):
            if not np.isfinite(v):
                continue
            t = float(np.clip(v, 1e-6, 1 - 1e-6))
            z = np.log(t / (1 - t))
            out[n] = float(1.0 / (1.0 + np.exp(-(a * z + b))))
        return out

    def available_predicates(self) -> List[str]:
        return self.rel.available_predicates()

    def set_predicates(self, names: Sequence[str]) -> List[str]:
        have = set(self.available_predicates())
        keep, seen = [], set()
        for n in names:
            n = n.strip()
            if n and n in have and n not in seen:
                keep.append(n); seen.add(n)
        dropped = [n for n in names if n.strip() and n.strip() not in have]
        if dropped:
            print(f"[engine] not in the predicate bank, skipped: {dropped}")
        if not keep:
            raise ValueError("none of the requested predicates are in the bank "
                             "(see --list-predicates)")
        self.rel.set_predicates(keep)
        self.thr_cfg.per_predicate = {n: self._bank_thr[n] for n in keep
                                      if n in self._bank_thr}
        return keep

    @property
    def predicates(self) -> List[str]:
        return list(self.rel.predicates)

    def spatial_map(self) -> Dict[str, bool]:
        is_sp = getattr(self.rel, "is_spatial", None)
        if is_sp is None:
            return {p: p in SPATIAL for p in self.predicates}
        return {p: bool(v) for p, v in zip(self.predicates,
                                           np.asarray(is_sp).ravel())}

    # -- inference ---------------------------------------------------------
    def __call__(self, frame_bgr: np.ndarray) -> FrameResult:
        from deploy.render_video import relate
        t0 = time.perf_counter()
        boxes, confs, labels = self.det(frame_bgr, self.det_cfg)
        t1 = time.perf_counter()
        triplets, raw = relate(self, frame_bgr, boxes, confs, labels)
        t2 = time.perf_counter()
        return FrameResult(boxes, confs, labels, triplets, raw,
                           (t1 - t0) * 1e3, (t2 - t1) * 1e3 if len(boxes) >= 2 else 0.0)

    def warmup(self) -> float:
        frame = (np.random.rand(360, 640, 3) * 255).astype(np.uint8)
        t = time.perf_counter()
        self.det(frame, self.det_cfg)
        boxes = np.array([[10, 10, 200, 300], [150, 100, 500, 340]], np.float32)
        self.rel(frame, boxes)
        return (time.perf_counter() - t) * 1e3
