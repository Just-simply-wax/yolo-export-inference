#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能对比脚本：同一批帧、同一套参数，比较 .pt / .onnx / .torchscript / .engine

流程（对应作业"对比"要求的每一条）：
  1) 从视频里均匀抽 N 帧（默认 73 帧），所有模型吃完全相同的数据；
  2) 每个模型先预热 warmup 次（排除加载/JIT/显存分配的一次性开销）；
  3) 重复 repeats 轮，逐帧计时，统计 平均/中位数/标准差/FPS；
  4) GPU 计时前后调用 torch.cuda.synchronize()，否则测到的只是"发指令"的时间；
  5) 以 .pt 为基线，比较检测数量、类别计数、框匹配率(IoU>0.5)、关键点平均像素误差；
  6) 输出 results/bench.json 与 results/comparison_auto.md（可直接粘进 README）。

用法：
    python benchmark.py --device cpu \
        --models models/best.onnx models/best.torchscript
    python benchmark.py --device cuda \
        --models models/best.onnx        # 需要 onnxruntime-gpu 才能用 GPU
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多格式推理性能对比")
    p.add_argument("--weights", default="weights/best.pt", help="基线模型（.pt）")
    p.add_argument("--models", nargs="*", default=["models/best.onnx", "models/best.torchscript"],
                   help="要对比的其它格式模型（可多个）")
    p.add_argument("--source", default="data/armor_raw.mp4", help="测试视频（与上次作业同一段）")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--frames", type=int, default=73, help="均匀抽样帧数")
    p.add_argument("--warmup", type=int, default=10, help="预热次数")
    p.add_argument("--repeats", type=int, default=3, help="测量轮数")
    p.add_argument("--device", default="cpu", help="cpu | cuda | 0")
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def read_frames(path: Path, n: int) -> list:
    """均匀抽取 n 帧，返回内存里的图像列表（保证所有模型用同一批图）。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频：{path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise SystemExit("读不到总帧数（视频可能损坏或编码不支持）")
    idxs = np.linspace(0, total - 1, min(n, total)).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit("没有取到任何帧")
    print(f"[info] 从 {path.name} 抽取 {len(frames)} 帧（总 {total} 帧）")
    return frames


def sync(device: str) -> None:
    """GPU 计时必须同步，否则异步执行会让耗时看起来"特别快"。"""
    if str(device).lower() == "cpu":
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def extract(result) -> dict:
    """把一帧的推理结果提取成 numpy，便于跨格式比较。"""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {"boxes": np.zeros((0, 4)), "confs": np.zeros((0,)),
                "clss": np.zeros((0,), dtype=int), "kpts": None}
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    kpts = None
    if result.keypoints is not None and len(result.keypoints.xy) > 0:
        kpts = result.keypoints.xy.cpu().numpy()
    return {"boxes": xyxy, "confs": confs, "clss": clss, "kpts": kpts}


def pair_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare(base: list, other: list):
    """逐帧对比：框匹配率（IoU>0.5 贪心匹配）与关键点平均像素误差。"""
    matched, base_total, kpt_errs = 0, 0, []
    for b, o in zip(base, other):
        bb, ob = b["boxes"], o["boxes"]
        base_total += len(bb)
        used = set()
        for i, bx in enumerate(bb):
            best_j, best_iou = -1, 0.0
            for j, ox in enumerate(ob):
                if j in used:
                    continue
                v = pair_iou(bx, ox)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0 and best_iou >= 0.5:
                used.add(best_j)
                matched += 1
                if b["kpts"] is not None and o["kpts"] is not None:
                    if i < len(b["kpts"]) and best_j < len(o["kpts"]):
                        d = np.linalg.norm(b["kpts"][i] - o["kpts"][best_j], axis=-1)
                        kpt_errs.append(float(d.mean()))
    rate = matched / base_total if base_total else 0.0
    return rate, (float(np.mean(kpt_errs)) if kpt_errs else None)


def bench(name: str, model_path: Path, frames: list, args, baseline: bool):
    print(f"\n===== {name} : {model_path} =====")
    if not model_path.exists():
        return {"name": name, "model": str(model_path), "error": "文件不存在"}
    size_mb = model_path.stat().st_size / 1024 / 1024

    try:
        model = YOLO(str(model_path))
        # ---- 预热 ----
        for k in range(args.warmup):
            model.predict(frames[k % len(frames)], imgsz=args.imgsz, conf=args.conf,
                          iou=args.iou, device=args.device, verbose=False)
        print(f"[info] 预热 {args.warmup} 次完成，开始测量 {args.repeats} 轮 × {len(frames)} 帧")

        times, dets = [], []
        t_all = time.perf_counter()
        for rep in range(args.repeats):
            for k, f in enumerate(frames):
                sync(args.device)
                t0 = time.perf_counter()
                r = model.predict(f, imgsz=args.imgsz, conf=args.conf,
                                  iou=args.iou, device=args.device, verbose=False)[0]
                sync(args.device)
                times.append((time.perf_counter() - t0) * 1000.0)
                if rep == 0:
                    dets.append(extract(r))
            print(f"  第 {rep + 1}/{args.repeats} 轮完成")
        wall = time.perf_counter() - t_all
    except Exception as e:
        return {"name": name, "model": str(model_path), "error": f"{type(e).__name__}: {e}"}

    counts, conf_sum, det_frames = Counter(), 0.0, 0
    names_map = model.names
    for d in dets:
        if len(d["clss"]):
            det_frames += 1
            conf_sum += float(d["confs"].sum())
            for c in d["clss"]:
                counts[names_map[int(c)]] += 1

    res = {
        "name": name,
        "model": str(model_path),
        "format": model_path.suffix.lstrip(".") if model_path.suffix.lower() != ".torchscript" else "torchscript",
        "size_mb": round(size_mb, 2),
        "device": args.device,
        "frames": len(frames) * args.repeats,
        "mean_ms": round(statistics.mean(times), 2),
        "median_ms": round(statistics.median(times), 2),
        "std_ms": round(statistics.pstdev(times), 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "fps": round(1000.0 / statistics.mean(times), 2) if times else 0,
        "wall_sec": round(wall, 2),
        "frames_with_detection": det_frames,
        "total_objects": sum(counts.values()),
        "per_class": dict(counts),
        "avg_conf": round(conf_sum / max(1, sum(counts.values())), 4),
        "is_baseline": baseline,
        "detections": dets,
    }
    print(f"[{name}] 平均 {res['mean_ms']} ms | 中位数 {res['median_ms']} ms | "
          f"std {res['std_ms']} ms | {res['fps']} FPS | 检出 {res['total_objects']} 个")
    return res


def md_table(rows: list, base_rate_map: dict) -> str:
    head = ("| 模型 | 格式 | 大小(MB) | 设备 | 平均(ms) | 中位数(ms) | 标准差 | FPS | "
            "检出数 | 框匹配率 | 关键点误差(px) |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    lines = [head]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['name']} | — | — | — | 失败 | — | — | — | — | — | {r['error']} |")
            continue
        rate = base_rate_map.get(r["name"], "")
        rate_s = "基线(100%)" if r["is_baseline"] else (f"{rate[0] * 100:.1f}%" if rate else "—")
        kpt_s = "0.00" if r["is_baseline"] else (f"{rate[1]:.2f}" if rate and rate[1] is not None else "—")
        lines.append(
            f"| {Path(r['model']).name} | {r['format']} | {r['size_mb']:.2f} | {r['device']} | "
            f"{r['mean_ms']:.2f} | {r['median_ms']:.2f} | {r['std_ms']:.2f} | {r['fps']:.2f} | "
            f"{r['total_objects']} | {rate_s} | {kpt_s} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"找不到测试视频：{src}")
    frames = read_frames(src, args.frames)

    results = []
    base = bench("baseline_pt", Path(args.weights), frames, args, baseline=True)
    results.append(base)
    for spec in args.models:
        # 允许写成 "标签=路径" 的形式，便于表格里显示可读名字
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            # 用带扩展名的文件名做标签：模型常同名（都叫 best.*），
            # 若用 .stem 会得到两个 "best"，导致一致性统计互相覆盖。
            label, path = Path(spec).name, spec
        results.append(bench(label, Path(path), frames, args, baseline=False))

    # 以基线为参照计算一致性
    rate_map = {}
    base_dets = base.get("detections") if "error" not in base else None
    if base_dets:
        for r in results:
            if r.get("is_baseline") or "error" in r:
                continue
            rate_map[r["name"]] = compare(base_dets, r["detections"])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table = md_table(results, rate_map)

    payload = [{k: v for k, v in r.items() if k != "detections"} for r in results]
    (outdir / "bench.json").write_text(
        json.dumps({"args": vars(args), "results": payload,
                    "consistency": {k: {"box_match_rate": v[0], "kpt_err_px": v[1]} for k, v in rate_map.items()}},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    md = (
        f"# 性能对比（自动生成）\n\n"
        f"- 测试视频：`{src.name}`\n"
        f"- 抽样帧数：{len(frames)}；预热：{args.warmup} 次；测量：{args.repeats} 轮\n"
        f"- 输入尺寸：{args.imgsz}；conf={args.conf}；iou={args.iou}；设备：{args.device}\n\n"
        f"{table}\n\n"
        f"> 说明：\"框匹配率\"= 以 .pt 为基线，IoU≥0.5 匹配上的框占基线框数的比例；"
        f"\"关键点误差\"= 匹配到同一目标后 4 个关键点的平均像素距离。\n"
    )
    (outdir / "comparison_auto.md").write_text(md, encoding="utf-8")

    print("\n" + table)
    print(f"\n[done] {outdir / 'bench.json'}\n[done] {outdir / 'comparison_auto.md'}")


if __name__ == "__main__":
    main()
