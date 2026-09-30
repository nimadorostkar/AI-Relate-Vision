#!/usr/bin/env python3
"""AI-Relate-Vision: real-time open-vocabulary relation prediction on your videos.

    ./run.sh                               # play every video in videos/ live
    ./run.sh videos/my_clip.mp4            # play one video live
    ./run.sh --export                      # render videos/* to outputs/ (mp4 + json)
    ./run.sh --webcam                      # live camera
    ./run.sh --predicates "holding, riding, looking at, sitting on"
    ./run.sh --list-predicates

(`./run.sh` is `.venv/bin/python relate_video.py`.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Real-time open-vocabulary relation prediction on your videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("inputs", nargs="*", default=[str(REPO / "videos")],
                    help="video files and/or folders")
    mode = ap.add_argument_group("mode")
    mode.add_argument("--export", action="store_true",
                      help="render annotated MP4 + JSON to --out instead of playing")
    mode.add_argument("--webcam", nargs="?", const=0, type=int, default=None,
                      metavar="INDEX", help="use a camera instead of files")
    mode.add_argument("--list-predicates", action="store_true",
                      help="print the relation phrases the model can score, and exit")
    mode.add_argument("--bench", action="store_true",
                      help="measure detector / relation latency on this machine, and exit")

    q = ap.add_argument_group("speed / quality")
    q.add_argument("--quality", default="balanced", choices=["fast", "balanced", "best"],
                   help="fast: det 416 + rel 256 px | balanced: 416 + 320 | best: 640 + 448")
    q.add_argument("--backend", default="openvino", choices=["openvino", "onnx"],
                   help="relation-head runtime (onnx always runs at 448 px)")
    q.add_argument("--threads", type=int, default=0, help="0 = physical CPU cores")
    q.add_argument("--width", type=int, default=960,
                   help="downscale frames to this long side (display + drawing)")
    q.add_argument("--stride", type=int, default=3,
                   help="--export: run the model every N frames")
    q.add_argument("--delay", type=float, default=0.8,
                   help="live: seconds of video decoded ahead of the screen, so results "
                        "land on the exact frame they were computed on")

    r = ap.add_argument_group("what to show")
    r.add_argument("--preset", default="all", choices=["all", "interaction", "spatial"])
    r.add_argument("--predicates", default="",
                   help="comma-separated relation phrases (see --list-predicates)")
    r.add_argument("--threshold", type=float, default=0.45,
                   help="calibrated relation confidence needed to show an edge")
    r.add_argument("--max-edges", type=int, default=8)
    r.add_argument("--max-spatial", type=int, default=3,
                   help="budget for spatial edges (on/behind/left of...); 0 hides them")
    r.add_argument("--det-conf", type=float, default=0.12,
                   help="detector confidence floor (497 classes -> low scores)")
    r.add_argument("--max-boxes", type=int, default=24)
    r.add_argument("--all-boxes", action="store_true",
                   help="also draw tracked objects that carry no relation")
    r.add_argument("--no-belief", action="store_true",
                   help="EMA smoothing instead of the Kalman relation belief")
    r.add_argument("--out", default=str(REPO / "outputs"), help="--export output folder")
    r.add_argument("--preview", action="store_true", help="--export: show a window too")
    r.add_argument("--hud", action="store_true", help="--export: burn timing into video")
    args = ap.parse_args()

    from relatevision.engine import EngineConfig, RelationEngine
    from relatevision.overlay import OverlayConfig

    preds = [p.strip() for p in args.predicates.split(",") if p.strip()] or None
    print("AI-Relate-Vision - Based on RelateAnything by Maëlic Neau "
          "(https://github.com/Maelic/RelateAnything), AGPL-3.0")
    print("[relate] loading models ...", flush=True)
    engine = RelationEngine(EngineConfig(
        quality=args.quality, backend=args.backend, threads=args.threads,
        det_conf=args.det_conf, max_boxes=args.max_boxes,
        predicates=preds, preset=args.preset))
    ms = engine.warmup()
    print(f"[relate] ready: relation head on {engine.backend} @ {engine.rel.img_size}px, "
          f"detector @ {engine.det.imgsz}px ({len(engine.det.classes)} object classes), "
          f"{engine.threads} threads, {len(engine.predicates)} predicates "
          f"(warmup {ms:.0f} ms)")

    if args.list_predicates:
        names = engine.available_predicates()
        print(f"{len(names)} relation phrases in the bank:")
        for i in range(0, len(names), 6):
            print("  " + " | ".join(names[i:i + 6]))
        return

    if args.bench:
        bench(engine)
        return

    ov_cfg = OverlayConfig(on_thr=args.threshold, max_edges=args.max_edges,
                           max_spatial=args.max_spatial, all_boxes=args.all_boxes,
                           belief=not args.no_belief)
    preset = "custom" if preds else args.preset

    from relatevision import player
    if args.webcam is not None:
        player.play_webcam(engine, args.webcam, ov_cfg, args.width, preset)
        return

    from relatevision.video import collect_inputs
    files = collect_inputs(args.inputs)
    if not files:
        sys.exit("no videos found - put .mp4/.mov files in the videos/ folder, "
                 "or pass a path: ./run.sh path/to/video.mp4")
    print(f"[relate] {len(files)} video(s): " + ", ".join(f.name for f in files))

    if args.export:
        for f in files:
            player.export(engine, f, Path(args.out), ov_cfg, args.width,
                          max(1, args.stride), args.preview, args.hud)
        return

    import cv2
    try:
        for f in files:
            if player.play_file(engine, f, ov_cfg, args.width, args.delay,
                                preset) == "quit":
                break
    finally:
        cv2.destroyAllWindows()


def bench(engine) -> None:
    import time
    import cv2
    import numpy as np
    img = cv2.imread(str(REPO / "assets/reel/images/horse.jpg"))
    img = cv2.resize(img, (960, int(960 * img.shape[0] / img.shape[1])))
    for _ in range(2):
        engine(img)
    det, rel = [], []
    for _ in range(8):
        r = engine(img)
        det.append(r.det_ms); rel.append(r.rel_ms)
    d, e = float(np.median(det)), float(np.median(rel))
    print(f"[bench] boxes={len(r.boxes)}  predicates={len(engine.predicates)}")
    print(f"  detector   {d:6.1f} ms")
    print(f"  relations  {e:6.1f} ms")
    print(f"  total      {d + e:6.1f} ms  -> {1000 / (d + e):.1f} model updates/s "
          "(video itself plays at full frame rate)")
    for t in r.triplets[:8]:
        print("   ", t)


if __name__ == "__main__":
    main()
