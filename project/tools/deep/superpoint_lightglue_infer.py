#!/usr/bin/env python3
"""运行 SuperPoint + LightGlue，并写出平台统一的 matches JSON。"""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--image1", required=True)
    parser.add_argument("--image2", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default="")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-matches", type=int, default=0)
    parser.add_argument("--resize", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        configure_ssl_certificates()
        add_default_third_party_path("LightGlue")
        import torch
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import load_image, rbd
    except Exception as exc:
        print(
            "SuperPoint+LightGlue inference requires Python packages: torch, lightglue. "
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 2

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor = SuperPoint(max_num_keypoints=args.max_matches if args.max_matches > 0 else 2048)
        matcher = LightGlue(features="superpoint")
        extractor = extractor.eval().to(device)
        matcher = matcher.eval().to(device)

        image0 = load_image(args.image1).to(device)
        image1 = load_image(args.image2).to(device)
        size0 = [int(image0.shape[-1]), int(image0.shape[-2])]
        size1 = [int(image1.shape[-1]), int(image1.shape[-2])]

        with torch.inference_mode():
            # SuperPoint 先提点和描述子，LightGlue 再输出匹配索引与分数。
            feats0 = extractor.extract(image0)
            feats1 = extractor.extract(image1)
            pred = matcher({"image0": feats0, "image1": feats1})

        feats0, feats1, pred = [rbd(x) for x in (feats0, feats1, pred)]
        matches = pred["matches"].detach().cpu()
        scores = pred.get("scores")
        scores = scores.detach().cpu() if scores is not None else torch.ones(matches.shape[0])
        kpts0 = feats0["keypoints"].detach().cpu()
        kpts1 = feats1["keypoints"].detach().cpu()

        rows = []
        for match, score in zip(matches, scores):
            c = float(score)
            if c < args.min_confidence:
                continue
            i0 = int(match[0])
            i1 = int(match[1])
            p0 = kpts0[i0]
            p1 = kpts1[i1]
            rows.append(
                {
                    "x1": float(p0[0]),
                    "y1": float(p0[1]),
                    "x2": float(p1[0]),
                    "y2": float(p1[1]),
                    "confidence": c,
                }
            )
        rows.sort(key=lambda item: item["confidence"], reverse=True)
        if args.max_matches > 0:
            rows = rows[: args.max_matches]

        output = {
            "method": "SUPERPOINT_LIGHTGLUE",
            "image1_size": size0,
            "image2_size": size1,
            "matches": rows,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:
        print(f"SuperPoint+LightGlue inference failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
