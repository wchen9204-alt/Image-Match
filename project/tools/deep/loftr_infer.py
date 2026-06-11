#!/usr/bin/env python3
"""运行 LoFTR，并写出平台统一的 matches JSON。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def configure_ssl_certificates() -> None:
    """优先使用 certifi 的 CA 证书，避免模型权重下载时证书链缺失。"""
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except Exception:
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image1", required=True)
    parser.add_argument("--image2", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default="")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-matches", type=int, default=0)
    parser.add_argument("--resize", type=int, default=0)
    return parser.parse_args()


def load_gray(path: str, resize: int):
    import cv2
    import torch

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    original_h, original_w = image.shape[:2]
    scale_x = 1.0
    scale_y = 1.0
    if resize > 0:
        # 只缩小长边，输出坐标再按比例映射回原图坐标系。
        long_side = max(original_w, original_h)
        if long_side > resize:
            scale = resize / float(long_side)
            new_w = max(1, int(round(original_w * scale)))
            new_h = max(1, int(round(original_h * scale)))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scale_x = original_w / float(new_w)
            scale_y = original_h / float(new_h)
    tensor = torch.from_numpy(image)[None, None].float() / 255.0
    return tensor, scale_x, scale_y, [original_w, original_h]


def main() -> int:
    args = parse_args()
    try:
        configure_ssl_certificates()
        import torch
        from kornia.feature import LoFTR
    except Exception as exc:
        print(
            "LoFTR inference requires Python packages: torch, kornia, opencv-python. "
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 2

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image0, sx0, sy0, size0 = load_gray(args.image1, args.resize)
        image1, sx1, sy1, size1 = load_gray(args.image2, args.resize)
        image0 = image0.to(device)
        image1 = image1.to(device)

        matcher = LoFTR(pretrained=None if args.weights else "outdoor").eval().to(device)
        if args.weights:
            checkpoint = torch.load(args.weights, map_location=device)
            state = checkpoint.get("state_dict", checkpoint)
            matcher.load_state_dict(state, strict=False)

        with torch.inference_mode():
            pred = matcher({"image0": image0, "image1": image1})

        # Kornia LoFTR 直接输出匹配后的坐标和置信度。
        kpts0 = pred["keypoints0"].detach().cpu().numpy()
        kpts1 = pred["keypoints1"].detach().cpu().numpy()
        conf = pred.get("confidence")
        conf = conf.detach().cpu().numpy() if conf is not None else [1.0] * len(kpts0)

        rows = []
        for p0, p1, c in zip(kpts0, kpts1, conf):
            c = float(c)
            if c < args.min_confidence:
                continue
            rows.append(
                {
                    "x1": float(p0[0] * sx0),
                    "y1": float(p0[1] * sy0),
                    "x2": float(p1[0] * sx1),
                    "y2": float(p1[1] * sy1),
                    "confidence": c,
                }
            )
        rows.sort(key=lambda item: item["confidence"], reverse=True)
        if args.max_matches > 0:
            rows = rows[: args.max_matches]

        output = {
            "method": "LOFTR",
            "image1_size": size0,
            "image2_size": size1,
            "matches": rows,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:
        print(f"LoFTR inference failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
