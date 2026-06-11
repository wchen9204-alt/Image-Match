#!/usr/bin/env python3
"""统一深度学习匹配 backend，支持 single 和 worker 两种执行模式。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
    """解析 C++ bridge 传入的统一 backend 参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "worker"], default="single")
    parser.add_argument("--method", required=True)
    parser.add_argument("--image1", default="")
    parser.add_argument("--image2", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--request-dir", default="")
    parser.add_argument("--weights", default="")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-matches", type=int, default=0)
    parser.add_argument("--resize", type=int, default=0)
    parser.add_argument("--poll-ms", type=int, default=50)
    return parser.parse_args()


def load_gray_tensor(path: str, resize: int):
    """读取灰度图并按长边缩放，返回 tensor、坐标回映射比例和原图尺寸。"""
    import cv2
    import torch

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")

    original_h, original_w = image.shape[:2]
    scale_x = 1.0
    scale_y = 1.0
    if resize > 0:
        # 只缩小长边，输出坐标再映射回原图坐标系，避免改变 C++ 几何估计坐标语义。
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


def write_json_atomic(path: Path, payload: dict) -> None:
    """原子写 JSON，避免 C++ 读取到半截响应或 matches 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict:
    """读取 request JSON。"""
    return json.loads(path.read_text(encoding="utf-8"))


class LoFTRBackend:
    """Kornia LoFTR backend，模型在 worker 生命周期内只初始化一次。"""

    def __init__(self, weights: str) -> None:
        import torch
        from kornia.feature import LoFTR

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.matcher = LoFTR(pretrained=None if weights else "outdoor").eval().to(self.device)
        if weights:
            checkpoint = torch.load(weights, map_location=self.device)
            state = checkpoint.get("state_dict", checkpoint)
            self.matcher.load_state_dict(state, strict=False)

    def infer(self, image1: str, image2: str, resize: int, min_confidence: float, max_matches: int) -> dict:
        """执行 LoFTR 推理并返回统一 matches payload。"""
        image0, sx0, sy0, size0 = load_gray_tensor(image1, resize)
        image1_tensor, sx1, sy1, size1 = load_gray_tensor(image2, resize)
        image0 = image0.to(self.device)
        image1_tensor = image1_tensor.to(self.device)

        with self.torch.inference_mode():
            # LoFTR 直接输出匹配后的两组坐标和置信度。
            pred = self.matcher({"image0": image0, "image1": image1_tensor})

        kpts0 = pred["keypoints0"].detach().cpu().numpy()
        kpts1 = pred["keypoints1"].detach().cpu().numpy()
        conf = pred.get("confidence")
        conf = conf.detach().cpu().numpy() if conf is not None else [1.0] * len(kpts0)
        rows = []
        for p0, p1, score in zip(kpts0, kpts1, conf):
            confidence = float(score)
            if confidence < min_confidence:
                continue
            rows.append(
                {
                    "x1": float(p0[0] * sx0),
                    "y1": float(p0[1] * sy0),
                    "x2": float(p1[0] * sx1),
                    "y2": float(p1[1] * sy1),
                    "confidence": confidence,
                }
            )
        return build_output("LOFTR", size0, size1, rows, max_matches)


class LightGlueBackend:
    """SuperPoint + LightGlue backend，模型在 worker 生命周期内只初始化一次。"""

    def __init__(self, max_matches: int) -> None:
        add_default_third_party_path("LightGlue")
        import torch
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import load_image, rbd

        self.torch = torch
        self.load_image = load_image
        self.rbd = rbd
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_keypoints = max_matches if max_matches > 0 else 2048
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def infer(self, image1: str, image2: str, resize: int, min_confidence: float, max_matches: int) -> dict:
        """执行 SuperPoint + LightGlue 推理并返回统一 matches payload。"""
        del resize  # LightGlue 官方工具保持原图坐标；该参数保留给统一接口。
        image0 = self.load_image(image1).to(self.device)
        image1_tensor = self.load_image(image2).to(self.device)
        size0 = [int(image0.shape[-1]), int(image0.shape[-2])]
        size1 = [int(image1_tensor.shape[-1]), int(image1_tensor.shape[-2])]

        with self.torch.inference_mode():
            # SuperPoint 先提点和描述子，LightGlue 再输出匹配索引与分数。
            feats0 = self.extractor.extract(image0)
            feats1 = self.extractor.extract(image1_tensor)
            pred = self.matcher({"image0": feats0, "image1": feats1})

        feats0, feats1, pred = [self.rbd(x) for x in (feats0, feats1, pred)]
        matches = pred["matches"].detach().cpu()
        scores = pred.get("scores")
        scores = scores.detach().cpu() if scores is not None else self.torch.ones(matches.shape[0])
        kpts0 = feats0["keypoints"].detach().cpu()
        kpts1 = feats1["keypoints"].detach().cpu()

        rows = []
        for match, score in zip(matches, scores):
            confidence = float(score)
            if confidence < min_confidence:
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
                    "confidence": confidence,
                }
            )
        return build_output("SUPERPOINT_LIGHTGLUE", size0, size1, rows, max_matches)


class SuperGlueBackend:
    """SuperPoint + SuperGlue backend，模型在 worker 生命周期内只初始化一次。"""

    def __init__(self, weights: str, min_confidence: float, max_matches: int) -> None:
        add_default_third_party_path("SuperGluePretrainedNetwork")
        import torch
        from models.matching import Matching

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_keypoints = max_matches if max_matches > 0 else 2048
        config = {
            "superpoint": {
                "nms_radius": 4,
                "keypoint_threshold": 0.005,
                "max_keypoints": max_keypoints,
            },
            "superglue": {
                "weights": self.weight_name(weights),
                "sinkhorn_iterations": 20,
                "match_threshold": max(0.0, min(1.0, min_confidence)),
            },
        }
        self.matcher = Matching(config).eval().to(self.device)

    @staticmethod
    def weight_name(value: str) -> str:
        """把配置中的权重字段转换为原版 SuperGlue 支持的 indoor/outdoor 名称。"""
        name = (value or "outdoor").strip().lower()
        if name in {"indoor", "outdoor"}:
            return name
        raise RuntimeError("SuperGlue weights must be 'indoor' or 'outdoor'.")

    def infer(self, image1: str, image2: str, resize: int, min_confidence: float, max_matches: int) -> dict:
        """执行 SuperPoint + SuperGlue 推理并返回统一 matches payload。"""
        image0, sx0, sy0, size0 = load_gray_tensor(image1, resize)
        image1_tensor, sx1, sy1, size1 = load_gray_tensor(image2, resize)
        image0 = image0.to(self.device)
        image1_tensor = image1_tensor.to(self.device)

        with self.torch.inference_mode():
            # 原版 Matching 同时执行 SuperPoint 提点和 SuperGlue 图匹配。
            pred = self.matcher({"image0": image0, "image1": image1_tensor})

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
            if confidence < min_confidence:
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
        return build_output("SUPERPOINT_SUPERGLUE", size0, size1, rows, max_matches)


def build_output(method: str, size0: list[int], size1: list[int], rows: list[dict], max_matches: int) -> dict:
    """统一排序、截断并生成 matches JSON payload。"""
    rows.sort(key=lambda item: item["confidence"], reverse=True)
    if max_matches > 0:
        rows = rows[:max_matches]
    return {
        "method": method,
        "image1_size": size0,
        "image2_size": size1,
        "matches": rows,
    }


def create_backend(method: str, weights: str, min_confidence: float, max_matches: int):
    """根据方法名创建具体 backend，worker 模式下只调用一次。"""
    key = method.strip().upper()
    if key == "LOFTR":
        return LoFTRBackend(weights)
    if key == "SUPERPOINT_LIGHTGLUE":
        return LightGlueBackend(max_matches)
    if key == "SUPERPOINT_SUPERGLUE":
        return SuperGlueBackend(weights, min_confidence, max_matches)
    raise RuntimeError(f"unsupported learning method: {method}")


def infer_to_file(backend, args: dict, default: argparse.Namespace) -> None:
    """执行一次推理请求，并把 matches JSON 写到 output 路径。"""
    image1 = args.get("image1") or default.image1
    image2 = args.get("image2") or default.image2
    output = args.get("output") or default.output
    if not image1 or not image2 or not output:
        raise RuntimeError("image1, image2 and output are required")

    min_confidence = float(args.get("min_confidence", default.min_confidence))
    max_matches = int(args.get("max_matches", default.max_matches))
    resize = int(args.get("resize", default.resize))
    payload = backend.infer(image1, image2, resize, min_confidence, max_matches)
    write_json_atomic(Path(output), payload)


def run_single(args: argparse.Namespace) -> int:
    """single 模式：加载一次模型、执行一次推理、写出 matches JSON 后退出。"""
    backend = create_backend(args.method, args.weights, args.min_confidence, args.max_matches)
    infer_to_file(backend, {}, args)
    return 0


def run_worker(args: argparse.Namespace) -> int:
    """worker 模式：加载一次模型，循环处理 C++ 写入的 request JSON。"""
    request_dir = Path(args.request_dir)
    if not request_dir:
        raise RuntimeError("--request-dir is required in worker mode")
    request_dir.mkdir(parents=True, exist_ok=True)
    backend = create_backend(args.method, args.weights, args.min_confidence, args.max_matches)
    print(f"Learning worker ready: method={args.method}, request_dir={request_dir}", flush=True)

    while True:
        # 1. stop 文件用于 C++ 进程正常退出时关闭 worker。
        if (request_dir / "stop").exists():
            return 0

        requests = sorted(request_dir.glob("request_*.json"))
        if not requests:
            time.sleep(max(1, args.poll_ms) / 1000.0)
            continue

        for request_path in requests:
            response_path = request_dir / request_path.name.replace("request_", "response_", 1)
            try:
                # 2. 每个 request 只描述一对图像和输出 JSON；模型对象在 backend 中复用。
                request = read_json(request_path)
                infer_to_file(backend, request, args)
                write_json_atomic(response_path, {"ok": True, "message": "OK"})
            except Exception as exc:
                write_json_atomic(response_path, {"ok": False, "message": str(exc)})
            finally:
                # 3. 删除已消费 request，避免 worker 重复处理同一任务。
                try:
                    request_path.unlink()
                except OSError:
                    pass


def main() -> int:
    args = parse_args()
    configure_ssl_certificates()
    try:
        if args.mode == "worker":
            return run_worker(args)
        return run_single(args)
    except Exception as exc:
        print(f"Learning backend failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
