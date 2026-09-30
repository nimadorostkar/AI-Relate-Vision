"""Convert the released relation ONNX graph to OpenVINO IR (run by setup_mac.sh).

    .venv/bin/python -m relatevision.convert

fp16 IR = compressed weights (half the file); on a CPU the math stays fp32, so
the decoded graph is unchanged. The engine reshapes the image input at load
time, so one IR serves every quality preset.
"""
from __future__ import annotations

import json
from pathlib import Path

DIST = Path(__file__).resolve().parents[1] / "deploy" / "dist" / "relsgg-vits16plus"


def main() -> None:
    import openvino as ov
    src = DIST / "relateanything.onnx"
    dst = DIST / "relateanything_fp16.xml"
    ov.save_model(ov.convert_model(str(src)), str(dst), compress_to_fp16=True)
    meta = json.loads((DIST / "relateanything.json").read_text())
    meta.update({"ov_variant": "fp16", "source_onnx": str(src.name)})
    dst.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"[convert] wrote {dst}")


if __name__ == "__main__":
    main()
