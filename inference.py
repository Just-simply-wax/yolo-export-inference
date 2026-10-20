#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一推理脚本：同一套代码加载 .pt / .onnx / .torchscript / .engine 并输出结果视频。

设计要点：
  - 由 ultralytics 的 AutoBackend 自动识别格式，绘图逻辑对所有格式完全一致，
    这样三种格式的截图才能并排对比。
  - 关键点来自 results[0].keypoints.xy（模型预测），不是检测框四角。

用法：
    python inference.py --model weights/best.pt          --device cuda --output results/pt/result_pt.mp4
    python inference.py --model models/best.onnx         --device cpu  --output results/onnx/result_onnx.mp4
    python inference.py --model models/best.torchscript  --device cpu  --output results/ts/result_ts.mp4
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PALETTE = [
    (0, 255, 0), (0, 128, 255), (255, 0, 0),
    (255, 0, 255), (0, 255, 255), (255, 255, 0),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="统一推理脚本（支持 pt / onnx / torchscript / engine）")
    p.add_argument("--model", default="weights/best.pt", help="模型文件（.pt/.onnx/.torchscript/.engine）")
    p.add_argument("--source", default="data/armor_raw.mp4", help="输入视频")
    p.add_argument("--output", default="results/pt/result.mp4", help="输出视频")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", default="cpu", help="cpu | cuda | 0")
    p.add_argument("--limit-frames", type=int, default=0, help="只处理前 N 帧（0=全部）")
    return p.parse_args()


def fourcc(code: str) -> int:
    if hasattr(cv2, "VideoWriter_fourcc"):
        return cv2.VideoWriter_fourcc(*code)
    return cv2.VideoWriter.fourcc(*code)


def pick_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def to_h264(tmp: Path, out: Path) -> bool:
    """OpenCV 只能方便地写 mp4v，转成 H.264/yuv420p 才能到处播放。"""
    ff = pick_ffmpeg()
    if ff is None:
        shutil.move(str(tmp), str(out))
        print("[warn] 未找到 ffmpeg，保留 mp4v 编码")
        return False
    subprocess.run(
        [ff, "-y", "-i", str(tmp),
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "-c:v", "libx264", "-preset", "medium", "-crf", "23",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    tmp.unlink(missing_ok=True)
    return True


def draw_detections(frame, result, names) -> tuple:
    """画框/类别/置信度/4 个关键点。三种格式共用，保证可视化一致。"""
    n_obj, n_kpt = 0, 0
    kpts = result.keypoints
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        cid, conf = int(box.cls.item()), float(box.conf.item())
        color = PALETTE[cid % len(PALETTE)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{names[cid]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        if kpts is not None and len(kpts.xy) > i:
            xy = kpts.xy[i].cpu().numpy()
            if len(xy) >= 4:
                n_kpt += 1
                cv2.polylines(frame, [xy.astype(np.int32)], True, color, 2, cv2.LINE_AA)
            for j, (px, py) in enumerate(xy):
                cv2.circle(frame, (int(px), int(py)), 4, color, -1, cv2.LINE_AA)
                cv2.putText(frame, f"P{j + 1}", (int(px) + 6, int(py) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        n_obj += 1
    return n_obj, n_kpt


def main() -> None:
    args = parse_args()
    m_path, s_path = Path(args.model), Path(args.source)
    if not m_path.exists():
        sys.exit(f"[error] 找不到模型：{m_path}")
    if not s_path.exists():
        sys.exit(f"[error] 找不到输入视频：{s_path}")

    model = YOLO(str(m_path))
    print(f"[info] 模型 {m_path.name}（{m_path.stat().st_size / 1024 / 1024:.2f} MB）task={model.task}")
    print(f"[info] 设备 {args.device} | imgsz={args.imgsz} conf={args.conf} iou={args.iou}")

    cap = cv2.VideoCapture(str(s_path))
    if not cap.isOpened():
        sys.exit(f"[error] 无法打开视频：{s_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 1:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[info] 视频 {w}x{h} @ {fps:.2f}fps，共 {total} 帧")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"_tmp_{out.stem}.mp4")
    writer = cv2.VideoWriter(str(tmp), fourcc("mp4v"), fps, (w, h))
    if not writer.isOpened():
        sys.exit("[error] VideoWriter 打开失败")

    counts = Counter()
    conf_sum = 0.0
    det_frames = 0
    idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                          device=args.device, verbose=False)[0]
        n_obj, _ = draw_detections(frame, r, model.names)
        if n_obj:
            det_frames += 1
            conf_sum += float(r.boxes.conf.sum().item())
            for c in r.boxes.cls.tolist():
                counts[model.names[int(c)]] += 1

        hud = f"{m_path.name} | {args.device} | frame {idx}/{total or '?'} | objects {n_obj}"
        cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)
        if idx % 60 == 0:
            print(f"  进度 {idx}/{total or '?'}")
        if args.limit_frames and idx >= args.limit_frames:
            break

    cap.release()
    writer.release()
    used = to_h264(tmp, out)

    elapsed = time.time() - t0
    stats = {
        "model": str(m_path),
        "model_mb": round(m_path.stat().st_size / 1024 / 1024, 2),
        "device": args.device,
        "imgsz": args.imgsz, "conf": args.conf, "iou": args.iou,
        "frames": idx, "frames_with_detection": det_frames,
        "total_objects": sum(counts.values()), "per_class": dict(counts),
        "avg_conf": round(conf_sum / max(1, sum(counts.values())), 4),
        "elapsed_sec": round(elapsed, 2),
        "avg_fps": round(idx / elapsed, 2) if elapsed else 0,
        "output": str(out), "output_h264": used,
    }
    print("[result] " + json.dumps(stats, ensure_ascii=False, indent=2))
    sp = out.with_suffix(".stats.json")
    sp.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] 结果视频：{out}\n[done] 统计：{sp}")


if __name__ == "__main__":
    main()
