#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 YOLO 的 .pt 权重导出为其他格式（作业要求：至少一种非 .pt 格式）。

支持：onnx / torchscript / engine(TensorRT) / openvino
导出产物统一挪到 models/ 目录，方便仓库管理。

用法：
    python export.py --weights weights/best.pt --format onnx --imgsz 640 --opset 12
    python export.py --weights weights/best.pt --format torchscript --imgsz 640
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from ultralytics import YOLO

# 各格式导出后常见的扩展名（用于把产物从原目录挪到 models/）
SUFFIX = {
    "onnx": [".onnx"],
    "torchscript": [".torchscript"],
    "engine": [".engine"],
    "openvino": [".xml", ".bin"],
}

# 每种格式"允许"接收的导出参数。
# ultralytics 会严格校验参数与格式的匹配关系，传了不支持的参数会直接 AssertionError，
# 例如：opset 只对 onnx 有意义，传给 torchscript 会报 "argument 'opset' is not supported"。
ALLOWED_ARGS = {
    "onnx": {"imgsz", "batch", "dynamic", "opset", "simplify", "half", "nms"},
    "torchscript": {"imgsz", "batch"},
    "engine": {"imgsz", "batch", "dynamic", "half", "simplify", "nms", "workspace"},
    "openvino": {"imgsz", "batch", "dynamic", "half", "nms", "int8"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="把 .pt 导出为其他推理格式")
    p.add_argument("--weights", default="weights/best.pt", help="原始 .pt 权重")
    p.add_argument("--format", default="onnx", choices=list(SUFFIX), help="目标格式")
    p.add_argument("--imgsz", type=int, default=640, help="导出时固定输入尺寸（要与推理时一致）")
    p.add_argument("--opset", type=int, default=12, help="ONNX 算子集版本")
    p.add_argument("--batch", type=int, default=1, help="静态批大小")
    p.add_argument("--dynamic", action="store_true", help="允许动态尺寸/批（默认静态，性能最好）")
    p.add_argument("--half", action="store_true", help="FP16（主要给 engine 用）")
    p.add_argument("--no-simplify", action="store_true", help="关闭 onnxslim 图简化")
    p.add_argument("--nms", action="store_true", help="把 NMS 融进模型（默认关闭，保持与 .pt 一致）")
    p.add_argument("--outdir", default="models", help="产物存放目录")
    return p.parse_args()


def inspect_onnx(path: Path) -> None:
    """打印 ONNX 的 opset 与输入/输出形状 —— 这是验证导出是否正确的第一手证据。"""
    try:
        import onnx
    except ImportError:
        print("[warn] 未安装 onnx，跳过结构检查（pip install onnx）")
        return
    m = onnx.load(str(path))
    print(f"[onnx] opset: {[(o.domain or 'ai.onnx', o.version) for o in m.opset_import]}")
    for i in m.graph.input:
        dims = [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]
        print(f"[onnx] 输入 {i.name}: {dims}")
    for o in m.graph.output:
        dims = [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
        print(f"[onnx] 输出 {o.name}: {dims}  ← 通道数 = 4(框) + 6(类别) + 12(4关键点×x,y,conf)")


def main() -> None:
    args = parse_args()
    w = Path(args.weights)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not w.exists():
        sys.exit(f"[error] 找不到权重文件：{w}")

    size_mb = w.stat().st_size / 1024 / 1024
    print(f"[info] 源权重: {w}  ({size_mb:.2f} MB)")

    t0 = time.time()
    model = YOLO(str(w))
    kpt = getattr(getattr(model, "model", None), "kpt_shape", None)
    print(f"[info] 模型: task={model.task} kpt_shape={kpt}")
    print(f"[info] 类别: {model.names}")

    # ---- 按格式筛选参数：只把该格式支持的参数传给 export() ----
    wanted = {
        "imgsz": args.imgsz,
        "batch": args.batch,
        "dynamic": args.dynamic,
        "opset": args.opset,
        "simplify": not args.no_simplify,
    }
    if args.half:
        wanted["half"] = True          # 只在用户显式要求时才传（该参数已标记 deprecated）
    if args.nms:
        wanted["nms"] = True           # pose 模型不支持融合 NMS，只在显式要求时才传

    allowed = ALLOWED_ARGS[args.format]
    kwargs = {k: v for k, v in wanted.items() if k in allowed}
    ignored = [k for k in wanted if k not in allowed]
    if ignored:
        print(f"[info] 格式 {args.format} 不支持这些参数，已自动忽略: {ignored}")

    exported = model.export(format=args.format, **kwargs)
    # ultralytics 的 export 返回 str 或 list[str]
    paths = exported if isinstance(exported, (list, tuple)) else [exported]

    moved = []
    for p in paths:
        src = Path(p)
        dst = outdir / src.name
        if src.resolve() != dst.resolve():
            shutil.move(str(src), str(dst))
        moved.append(dst)

    print("\n========== 导出结果 ==========")
    for p in moved:
        print(f"[done] {p}  ({p.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"[done] 用时 {time.time() - t0:.1f} 秒，格式={args.format}，imgsz={args.imgsz}")

    if args.format == "onnx":
        inspect_onnx(moved[0])

    # 提醒：体积对比与推理验证
    print("\n[下一步] 验证能推理：")
    print(f"  python -c \"from ultralytics import YOLO;"
          f"m=YOLO('{moved[0]}');r=next(iter(m.predict('data/armor_raw.mp4',"
          f"imgsz={args.imgsz},conf=0.25,device='cpu',stream=True,verbose=False)));"
          f"print(len(r.boxes), None if r.keypoints is None else tuple(r.keypoints.xy.shape))\"")


if __name__ == "__main__":
    main()
