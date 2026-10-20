#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证导出模型能否推理（避免在终端里跟引号较劲）。

一次检查多个模型：.pt（基线）、.onnx、.torchscript、.engine，
打印各自的 task / 文件大小 / 检出框数 / 关键点形状，并对比框数是否与基线一致。

用法：
    python verify_export.py                       # 默认检查 pt + onnx + torchscript
    python verify_export.py --models models/best.onnx
    python verify_export.py --source data/armor_raw.mp4 --imgsz 640 --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="验证导出模型能否正常推理")
    p.add_argument("--models", nargs="*",
                   default=["weights/best.pt", "models/best.onnx", "models/best.torchscript"],
                   help="要检查的模型路径（可多个）")
    p.add_argument("--source", default="data/armor_raw.mp4", help="测试输入（视频或图片）")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="cpu", help="cpu | cuda | 0")
    return p.parse_args()


def inspect_onnx_struct(path: Path) -> None:
    """打印 ONNX 的 opset 与输入/输出形状（只在装了 onnx 时执行）。"""
    try:
        import onnx
    except ImportError:
        return
    m = onnx.load(str(path))
    opset = [(o.domain or "ai.onnx", o.version) for o in m.opset_import]
    print(f"    opset={opset}")
    for i in m.graph.input:
        print(f"    输入 {i.name}: {[d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]}")
    for o in m.graph.output:
        print(f"    输出 {o.name}: {[d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]}")


def check_one(path: Path, args) -> dict | None:
    if not path.exists():
        print(f"[跳过] 文件不存在：{path}")
        return None

    print(f"\n===== {path} =====")
    print(f"    大小: {path.stat().st_size / 1024 / 1024:.2f} MB")
    try:
        model = YOLO(str(path))
        r = next(iter(model.predict(args.source, imgsz=args.imgsz, conf=args.conf,
                                    device=args.device, stream=True, verbose=False)))
    except Exception as e:
        print(f"[失败] 加载或推理出错：{type(e).__name__}: {e}")
        return None

    n_box = len(r.boxes)
    kpt_shape = None if r.keypoints is None else tuple(r.keypoints.xy.shape)
    kpt_ok = bool(kpt_shape and len(kpt_shape) == 3 and kpt_shape[1] == 4 and kpt_shape[2] == 2)

    print(f"    task={model.task}  类别数={len(model.names)}")
    print(f"    检出框数={n_box}  关键点形状={kpt_shape}  "
          f"{'✅ 4 关键点正常' if kpt_ok else '⚠️ 关键点异常（应为 (N, 4, 2)）'}")

    if path.suffix == ".onnx":
        inspect_onnx_struct(path)

    return {
        "name": path.name,
        "boxes": n_box,
        "kpt_shape": kpt_shape,
        "cls": r.boxes.cls.cpu().numpy().astype(int).tolist() if n_box else [],
    }


def main() -> None:
    args = parse_args()
    if not Path(args.source).exists():
        sys.exit(f"[error] 找不到测试输入：{args.source}")

    print(f"[info] 输入={args.source}  imgsz={args.imgsz}  conf={args.conf}  device={args.device}")
    results = [check_one(Path(m), args) for m in args.models]
    results = [r for r in results if r]

    if len(results) < 2:
        print("\n[提示] 至少要有两个成功加载的模型才能做对比。")
        return

    base = results[0]
    print("\n========== 一致性小结（以第一个模型 " + base["name"] + " 为基线） ==========")
    print(f"{'模型':<22}{'框数':>6}{'类别一致':>10}{'关键点':>16}")
    for r in results:
        same = "—" if r is base else ("是" if r["cls"] == base["cls"] else "否")
        print(f"{r['name']:<22}{r['boxes']:>6}{same:>10}{str(r['kpt_shape']):>16}")

    print("\n说明：同一帧上，导出模型与 .pt 的框数、类别应当一致；"
          "若不同，先确认 imgsz / conf 是否完全相同。")


if __name__ == "__main__":
    main()
