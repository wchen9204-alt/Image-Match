#!/usr/bin/env python3
"""运行 SuperPoint + SuperGlue，并写出平台统一的 matches JSON。"""

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


def add_default_third_party_path(repo_name: str) -> None:
    """把项目内 third_party 依赖仓库加入 sys.path，便于离线源码方式运行。"""
    repo = Path(__file__).resolve().parents[2] / "third_party" / repo_name
    if repo.exists() and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def parse_args() -> argparse.Namespace:
    """解析 C++ bridge 传入的通用深度匹配参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image1", required=True)
    parser.add_argument("--image2", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default="outdoor")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-matches", type=int, default=0)
    parser.add_argument("--resize", type=int, default=0)
    return parser.parse_args()


def load_gray_tensor(path: str, resize: int):
    """读取灰度图并转为 SuperGlue 原版实现需要的 1x1xHxW tensor。"""
    import cv2
    import torch

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")

    original_h, original_w = image.shape[:2]
    scale_x = 1.0
    scale_y = 1.0
    if resize > 0:
        # 只缩小长边，避免推理显存过高；输出坐标会映射回原图坐标系。
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


def superglue_weight_name(value: str) -> str:
    """把配置中的权重字段转换为原版 SuperGlue 支持的 indoor/outdoor 名称。"""
    name = (value or "outdoor").strip().lower()
    if name in {"indoor", "outdoor"}:
        return name
    raise RuntimeError(
        "SuperGlue weights must be 'indoor' or 'outdoor' for the original "
        "SuperGluePretrainedNetwork implementation."
    )


def main() -> int:
    args = parse_args()
    try:
        configure_ssl_certificates()
        add_default_third_party_path("SuperGluePretrainedNetwork")
        import torch
        from models.matching import Matching
    except Exception as exc:
        print(
            "SuperPoint+SuperGlue inference requires Python packages: torch, opencv-python, "
            "and MagicLeap SuperGluePretrainedNetwork on PYTHONPATH. "
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 2

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image0, sx0, sy0, size0 = load_gray_tensor(args.image1, args.resize)
        image1, sx1, sy1, size1 = load_gray_tensor(args.image2, args.resize)
        image0 = image0.to(device)
        image1 = image1.to(device)

        # SuperGlue 原版配置中 max_keypoints 控制 SuperPoint 提点数量，match_threshold 控制匹配置信度。
        config = {
            "superpoint": {
                "nms_radius": 4,
                "keypoint_threshold": 0.005,
                "max_keypoints": args.max_matches if args.max_matches > 0 else 2048,
            },
            "superglue": {
                "weights": superglue_weight_name(args.weights),
                "sinkhorn_iterations": 20,
                "match_threshold": max(0.0, min(1.0, args.min_confidence)),
            },
        }
        matcher = Matching(config).eval().to(device)

        with torch.inference_mode():
            # 原版 Matching 同时执行 SuperPoint 提点和 SuperGlue 图匹配。
            pred = matcher({"image0": image0, "image1": image1})

        kpts0 = pred["keypoints0"][0].detach().cpu()
        kpts1 = pred["keypoints1"][0].detach().cpu()
        matches0 = pred["matches0"][0].detach().cpu()
        scores0 = pred["matching_scores0"][0].detach().cpu()

        rows = []
        for i0, i1_tensor in enumerate(matches0):
            i1 = int(i1_tensor)
            if i1 < 0:
                continue
            confidence = float(scores0[i0])
            if confidence < args.min_confidence:
                continue
            p0 = kpts0[i0]
            p1 = kpts1[i1]
            rows.append(
                {
                    "x1": float(p0[0] * sx0),
                    "y1": float(p0[1] * sy0),
                    "x2": float(p1[0] * sx1),
                    "y2": float(p1[1] * sy1),
                    "confidence": confidence,
                }
            )

        rows.sort(key=lambda item: item["confidence"], reverse=True)
        if args.max_matches > 0:
            rows = rows[: args.max_matches]

        output = {
            "method": "SUPERPOINT_SUPERGLUE",
            "image1_size": size0,
            "image2_size": size1,
            "matches": rows,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:
        print(f"SuperPoint+SuperGlue inference failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
