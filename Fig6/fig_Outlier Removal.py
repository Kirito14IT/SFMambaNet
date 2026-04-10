#!/usr/bin/env python3
"""
Beginner-friendly Fig.6-style visualization script.

Main mode:
  Input: two image paths + one method
  Output: one left-right correspondence visualization

Helper modes:
  --list-scenes
  --list-images
  --list-pairs

Notes:
- Default output format is PDF.
- The background photos remain raster images, while lines are vector
  objects in PDF/EPS output. This is normal for paper figures.
- Supported methods in the current repo: input, oanet, lfgc.
- You can also load a custom checkpoint with:
  --method custom --module-path xxx.py --class-name XxxNet --checkpoint xxx.pth
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)


METHOD_SPECS = {
    "oanet": {
        "title": "OANet",
        "module_path": os.path.join(SCRIPT_DIR, "oan文件", "oan.py"),
        "checkpoint_template": os.path.join(
            SCRIPT_DIR, "model", "{dataset}", "essential", "sift-2000", "model_best.pth"
        ),
    },
    "lfgc": {
        "title": "LFGC",
        "module_path": os.path.join(SCRIPT_DIR, "oanlfgc.py"),
        "checkpoint_template": os.path.join(
            SCRIPT_DIR, "1LFGC", "{dataset}", "model_best.pth"
        ),
    },
}

VALID_DATASETS = ("yfcc", "sun3d")


def normalize_method(method: str) -> str:
    alias = {
        "oa": "oanet",
        "oanet": "oanet",
        "lfgc": "lfgc",
        "input": "input",
        "custom": "custom",
    }
    key = method.lower()
    if key not in alias:
        raise ValueError(f"Unsupported method: {method}")
    return alias[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Fig.6-style two-view match visualization from two image paths."
    )

    parser.add_argument("--img1", type=str, help="Path of the left image.")
    parser.add_argument("--img2", type=str, help="Path of the right image.")
    parser.add_argument(
        "--method",
        type=str,
        default="input",
        help="input / oanet / lfgc / custom. Alias oa is also supported.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. For built-in methods it overrides the default checkpoint. For custom method it is required.",
    )
    parser.add_argument(
        "--module-path",
        type=str,
        default=None,
        help="Optional Python module path that defines the model class. Required for --method custom.",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default=None,
        help="Optional model class name inside --module-path, for example BCLNet or MatchMamba.",
    )
    parser.add_argument(
        "--checkpoint-key",
        type=str,
        default="auto",
        help="How to read weights from the checkpoint. Default auto tries common keys such as state_dict.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model-threshold", type=float, default=0.0)
    parser.add_argument("--geod-threshold", type=float, default=1e-4)
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Optional max number of lines to draw. Useful when the image is too dense.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--line-width", type=float, default=1.2)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--panel-size", type=float, default=5.8)
    parser.add_argument(
        "--output-format",
        type=str,
        default="pdf",
        choices=["pdf", "eps", "png"],
        help="Default is pdf.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file path. If omitted, an automatic name is used.",
    )

    parser.add_argument("--dataset", type=str, choices=VALID_DATASETS, help="Used by helper modes.")
    parser.add_argument("--scene", type=str, help="Used by helper modes.")
    parser.add_argument("--limit", type=int, default=10, help="Helper mode print limit.")
    parser.add_argument("--list-scenes", action="store_true", help="List available scenes.")
    parser.add_argument("--list-images", action="store_true", help="List images in one scene.")
    parser.add_argument("--list-pairs", action="store_true", help="List image pairs in one scene.")

    args = parser.parse_args()
    args.method = normalize_method(args.method)
    return args


def resolve_input_path(path_str: str) -> str:
    candidates = [
        path_str,
        os.path.join(REPO_ROOT, path_str),
        os.path.join(SCRIPT_DIR, path_str),
    ]
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Path not found: {path_str}")


def resolve_optional_path(path_str: Optional[str]) -> Optional[str]:
    if path_str is None:
        return None
    return resolve_input_path(path_str)


def load_module(module_name: str, module_path: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_model_config():
    config_dict = {
        "net_channels": 128,
        "net_depth": 12,
        "clusters": 500,
        "use_ratio": 0,
        "use_mutual": 0,
        "iter_num": 1,
    }
    return namedtuple("Config", config_dict.keys())(*config_dict.values())


def load_checkpoint_file(checkpoint_path: str, device: torch.device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def extract_state_dict(checkpoint_obj, checkpoint_key: str):
    if checkpoint_key != "auto":
        if not isinstance(checkpoint_obj, dict):
            raise ValueError("checkpoint_key was specified, but the loaded checkpoint is not a dict.")
        if checkpoint_key not in checkpoint_obj:
            raise KeyError(f"checkpoint key '{checkpoint_key}' not found.")
        return checkpoint_obj[checkpoint_key]

    if isinstance(checkpoint_obj, dict):
        preferred_keys = [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
        ]
        for key in preferred_keys:
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                return checkpoint_obj[key]

        if checkpoint_obj and all(hasattr(v, "shape") for v in checkpoint_obj.values()):
            return checkpoint_obj

    return checkpoint_obj


def infer_class_name(module, requested_class_name: Optional[str]) -> str:
    if requested_class_name:
        if not hasattr(module, requested_class_name):
            raise AttributeError(f"Class '{requested_class_name}' not found in module.")
        return requested_class_name

    for candidate in ["OANet", "LFGC", "BCLNet", "MatchMamba", "SFMambaNet", "Model", "Net"]:
        if hasattr(module, candidate):
            return candidate

    raise ValueError(
        "Failed to infer model class name automatically. Please pass --class-name explicitly."
    )


def instantiate_model_from_module(module, class_name: str):
    model_cls = getattr(module, class_name)
    config = default_model_config()

    constructor_attempts = [
        (config,),
        tuple(),
    ]
    errors = []
    for args in constructor_attempts:
        try:
            return model_cls(*args)
        except TypeError as exc:
            errors.append(str(exc))

    raise TypeError(
        f"Failed to construct model class '{class_name}'. "
        f"Tried constructors ({class_name}(config), {class_name}()). "
        f"Errors: {' | '.join(errors)}"
    )


def load_h5(path: str) -> Dict:
    with h5py.File(path, "r") as h5file:
        def read_node(node):
            data = {}
            for key in node.keys():
                item = node[key]
                if isinstance(item, h5py.Group):
                    data[key] = read_node(item)
                else:
                    data[key] = item[()]
            return data

        return read_node(h5file)


def dataset_root(dataset: str) -> str:
    return os.path.join(SCRIPT_DIR, dataset)


def raw_test_root(dataset: str, scene: str) -> str:
    return os.path.join(dataset_root(dataset), "raw_data", scene, "test")


def dump_root(dataset: str, scene: str) -> str:
    return os.path.join(dataset_root(dataset), "data_dump", scene, "sift-2000", "test", "dump")


def list_scenes() -> None:
    for dataset in VALID_DATASETS:
        root = os.path.join(dataset_root(dataset), "raw_data")
        if not os.path.isdir(root):
            continue
        scenes = sorted(
            item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))
        )
        print(f"\n[{dataset}]")
        for scene in scenes:
            print(f"  {scene}")


def read_images_txt(dataset: str, scene: str) -> List[str]:
    image_txt = os.path.join(raw_test_root(dataset, scene), "images.txt")
    if not os.path.isfile(image_txt):
        raise FileNotFoundError(f"images.txt not found: {image_txt}")
    with open(image_txt, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def list_images(dataset: str, scene: str, limit: int) -> None:
    images = read_images_txt(dataset, scene)
    print(f"\n[{dataset}/{scene}] images (first {min(limit, len(images))})")
    for idx, rel_path in enumerate(images[:limit]):
        abs_path = os.path.abspath(os.path.join(raw_test_root(dataset, scene), rel_path))
        print(f"{idx:>4}: {abs_path}")


def parse_pair_name(pair_name: str) -> Tuple[int, int]:
    parts = pair_name.split("-")
    if len(parts) != 3 or not parts[2].endswith(".h5"):
        raise ValueError(f"Unexpected pair file name: {pair_name}")
    return int(parts[1]), int(parts[2].replace(".h5", ""))


def list_pairs(dataset: str, scene: str, limit: int) -> None:
    images = read_images_txt(dataset, scene)
    dump_dir = dump_root(dataset, scene)
    if not os.path.isdir(dump_dir):
        raise FileNotFoundError(f"Pair folder not found: {dump_dir}")
    pair_names = sorted(
        item for item in os.listdir(dump_dir) if item.endswith(".h5")
    )
    print(f"\n[{dataset}/{scene}] pairs (first {min(limit, len(pair_names))})")
    for idx, pair_name in enumerate(pair_names[:limit], start=1):
        img1_id, img2_id = parse_pair_name(pair_name)
        img1_abs = os.path.abspath(os.path.join(raw_test_root(dataset, scene), images[img1_id]))
        img2_abs = os.path.abspath(os.path.join(raw_test_root(dataset, scene), images[img2_id]))
        print(f"{idx:>3}. {pair_name}")
        print(f"     img1: {img1_abs}")
        print(f"     img2: {img2_abs}")


def infer_dataset_scene(abs_image_path: str) -> Tuple[str, str]:
    parts = os.path.normpath(abs_image_path).split(os.sep)
    for idx, part in enumerate(parts):
        if part in VALID_DATASETS:
            if idx + 4 >= len(parts):
                continue
            if parts[idx + 1] == "raw_data" and parts[idx + 3] == "test" and parts[idx + 4] == "images":
                dataset = part
                scene = parts[idx + 2]
                return dataset, scene
    raise ValueError(
        "Failed to infer dataset/scene from image path. "
        "The image must be under quankeshihua/<dataset>/raw_data/<scene>/test/images/."
    )


def build_scene_index(dataset: str, scene: str) -> Dict:
    images = read_images_txt(dataset, scene)
    raw_root = raw_test_root(dataset, scene)
    rel_to_id = {}
    abs_to_id = {}
    for idx, rel_path in enumerate(images):
        rel_norm = os.path.normpath(rel_path)
        abs_path = os.path.abspath(os.path.join(raw_root, rel_norm))
        rel_to_id[rel_norm] = idx
        abs_to_id[os.path.normcase(abs_path)] = idx
    return {
        "images": images,
        "raw_root": raw_root,
        "rel_to_id": rel_to_id,
        "abs_to_id": abs_to_id,
    }


def find_pair_from_images(img1_path: str, img2_path: str) -> Dict:
    abs_img1 = resolve_input_path(img1_path)
    abs_img2 = resolve_input_path(img2_path)

    dataset1, scene1 = infer_dataset_scene(abs_img1)
    dataset2, scene2 = infer_dataset_scene(abs_img2)
    if dataset1 != dataset2 or scene1 != scene2:
        raise ValueError("The two images must come from the same dataset and the same scene.")

    dataset = dataset1
    scene = scene1
    scene_index = build_scene_index(dataset, scene)
    norm_img1 = os.path.normcase(abs_img1)
    norm_img2 = os.path.normcase(abs_img2)

    if norm_img1 not in scene_index["abs_to_id"]:
        raise ValueError(f"Image not listed in images.txt: {abs_img1}")
    if norm_img2 not in scene_index["abs_to_id"]:
        raise ValueError(f"Image not listed in images.txt: {abs_img2}")

    img1_id = scene_index["abs_to_id"][norm_img1]
    img2_id = scene_index["abs_to_id"][norm_img2]
    pair_dir = dump_root(dataset, scene)

    direct_pair = f"nn-{img1_id}-{img2_id}.h5"
    reverse_pair = f"nn-{img2_id}-{img1_id}.h5"
    direct_path = os.path.join(pair_dir, direct_pair)
    reverse_path = os.path.join(pair_dir, reverse_pair)

    if os.path.isfile(direct_path):
        return {
            "dataset": dataset,
            "scene": scene,
            "pair_name": direct_pair,
            "pair_order_matches_input": True,
            "requested_img1": abs_img1,
            "requested_img2": abs_img2,
        }

    if os.path.isfile(reverse_path):
        return {
            "dataset": dataset,
            "scene": scene,
            "pair_name": reverse_pair,
            "pair_order_matches_input": False,
            "requested_img1": abs_img1,
            "requested_img2": abs_img2,
        }

    raise FileNotFoundError(
        "No nn-*.h5 pair file was found for the selected image pair. "
        "Choose two images that appear in the same available pair."
    )


def calibration_name_from_index(image_index: int) -> str:
    return f"calibration_{image_index + 1:06d}.h5"


def np_skew_symmetric(v: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(v[:, 0])
    return np.stack(
        [
            zero,
            -v[:, 2],
            v[:, 1],
            v[:, 2],
            zero,
            -v[:, 0],
            -v[:, 1],
            v[:, 0],
            zero,
        ],
        axis=1,
    )


def get_episym(x1: np.ndarray, x2: np.ndarray, dR: np.ndarray, dt: np.ndarray) -> np.ndarray:
    num_pts = len(x1)
    x1 = np.concatenate([x1, np.ones((num_pts, 1))], axis=-1).reshape(-1, 3, 1)
    x2 = np.concatenate([x2, np.ones((num_pts, 1))], axis=-1).reshape(-1, 3, 1)

    dR = dR.reshape(1, 3, 3)
    dt = dt.reshape(1, 3)
    F = np.repeat(
        np.matmul(np.reshape(np_skew_symmetric(dt), (-1, 3, 3)), dR).reshape(-1, 3, 3),
        num_pts,
        axis=0,
    )

    x2Fx1 = np.matmul(x2.transpose(0, 2, 1), np.matmul(F, x1)).flatten()
    Fx1 = np.matmul(F, x1).reshape(-1, 3)
    Ftx2 = np.matmul(F.transpose(0, 2, 1), x2).reshape(-1, 3)
    return x2Fx1 ** 2 * (
        1.0 / (Fx1[..., 0] ** 2 + Fx1[..., 1] ** 2)
        + 1.0 / (Ftx2[..., 0] ** 2 + Ftx2[..., 1] ** 2)
    )


def parse_geom(geom_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    K = geom_dict["K"].reshape(3, 3).astype(np.float64)
    R = geom_dict["R"].reshape(3, 3).astype(np.float64)
    t = geom_dict["T"].reshape(3, 1).astype(np.float64)
    img_size = geom_dict["imsize"].reshape(2).astype(np.float64)
    return {
        "K": K,
        "R": R,
        "t": t,
        "img_size": img_size,
        "K_inv": np.linalg.inv(K),
    }


def unpack_K(geom: Dict[str, np.ndarray]) -> Tuple[float, float, List[float]]:
    img_size, K = geom["img_size"], geom["K"]
    w, h = img_size[0], img_size[1]
    cx = (w - 1.0) * 0.5 + K[0, 2]
    cy = (h - 1.0) * 0.5 + K[1, 2]
    return float(cx), float(cy), [float(K[0, 0]), float(K[1, 1])]


def norm_kp(cx: float, cy: float, fx: float, fy: float, kp: np.ndarray) -> np.ndarray:
    return (kp - np.array([[cx, cy]], dtype=np.float64)) / np.asarray([[fx, fy]], dtype=np.float64)


def load_scene_pair_from_images(img1_path: str, img2_path: str) -> Dict:
    pair_info = find_pair_from_images(img1_path, img2_path)
    dataset = pair_info["dataset"]
    scene = pair_info["scene"]
    pair_name = pair_info["pair_name"]
    raw_root = raw_test_root(dataset, scene)
    image_list = read_images_txt(dataset, scene)
    img1_id, img2_id = parse_pair_name(pair_name)

    img1_rel = image_list[img1_id]
    img2_rel = image_list[img2_id]
    img1_abs = os.path.abspath(os.path.join(raw_root, img1_rel))
    img2_abs = os.path.abspath(os.path.join(raw_root, img2_rel))

    image0 = cv2.imread(img1_abs)
    image1 = cv2.imread(img2_abs)
    if image0 is None or image1 is None:
        raise FileNotFoundError("Failed to read one of the input images from the pair file.")
    image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    kp1_path = os.path.join(raw_root, img1_rel + ".sift-2000.hdf5")
    kp2_path = os.path.join(raw_root, img2_rel + ".sift-2000.hdf5")
    cal1_path = os.path.join(raw_root, "calibration", calibration_name_from_index(img1_id))
    cal2_path = os.path.join(raw_root, "calibration", calibration_name_from_index(img2_id))
    pair_path = os.path.join(dump_root(dataset, scene), pair_name)

    kp_i = load_h5(kp1_path)["keypoints"][:, :2].astype(np.float64)
    kp_j = load_h5(kp2_path)["keypoints"][:, :2].astype(np.float64)
    geom1 = parse_geom(load_h5(cal1_path))
    geom2 = parse_geom(load_h5(cal2_path))
    idx_sort = load_h5(pair_path)["idx_sort"].astype(np.int64)

    cx1, cy1, f1 = unpack_K(geom1)
    cx2, cy2, f2 = unpack_K(geom2)
    x1 = norm_kp(cx1, cy1, f1[0], f1[1], kp_i)
    x2 = norm_kp(cx2, cy2, f2[0], f2[1], kp_j)
    x2_ordered = x2[idx_sort[1], :]

    R_i, R_j = geom1["R"], geom2["R"]
    dR = np.dot(R_j, R_i.T)
    t_i, t_j = geom1["t"].reshape(3, 1), geom2["t"].reshape(3, 1)
    dt = t_j - np.dot(dR, t_i)
    dt = dt / np.sqrt(np.sum(dt ** 2))

    return {
        "dataset": dataset,
        "scene": scene,
        "pair_name": pair_name,
        "pair_order_matches_input": pair_info["pair_order_matches_input"],
        "requested_img1": pair_info["requested_img1"],
        "requested_img2": pair_info["requested_img2"],
        "pair_img1": img1_abs,
        "pair_img2": img2_abs,
        "image0": image0,
        "image1": image1,
        "kp_i": kp_i,
        "kp_j": kp_j,
        "x1": x1,
        "x2": x2,
        "x2_ordered": x2_ordered,
        "idx_sort": idx_sort,
        "dR": dR,
        "dt": dt,
        "xs": np.concatenate([x1, x2_ordered], axis=1).reshape(1, -1, 4).astype(np.float32),
    }


def build_model(
    method: str,
    dataset: str,
    device: torch.device,
    checkpoint_override: Optional[str] = None,
    module_path_override: Optional[str] = None,
    class_name_override: Optional[str] = None,
    checkpoint_key: str = "auto",
):
    if method == "input":
        return None

    if method == "custom":
        if checkpoint_override is None:
            raise ValueError("--method custom requires --checkpoint.")
        if module_path_override is None:
            raise ValueError("--method custom requires --module-path.")
        checkpoint_path = resolve_input_path(checkpoint_override)
        module_path = resolve_input_path(module_path_override)
    else:
        spec = METHOD_SPECS[method]
        checkpoint_path = (
            resolve_input_path(checkpoint_override)
            if checkpoint_override is not None
            else spec["checkpoint_template"].format(dataset=dataset)
        )
        module_path = (
            resolve_input_path(module_path_override)
            if module_path_override is not None
            else spec["module_path"]
        )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"Module path not found: {module_path}")

    module = load_module(f"{method}_module", module_path)
    class_name = infer_class_name(module, class_name_override)
    model = instantiate_model_from_module(module, class_name)
    checkpoint = load_checkpoint_file(checkpoint_path, device)
    state_dict = extract_state_dict(checkpoint, checkpoint_key)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def select_matches(
    method: str,
    model,
    scene_data: Dict,
    device: torch.device,
    model_threshold: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if method == "input":
        selected = np.arange(scene_data["idx_sort"].shape[1], dtype=np.int64)
        return selected, None

    with torch.no_grad():
        data = {"xs": torch.from_numpy(scene_data["xs"]).unsqueeze(0).to(device)}
        y_hat, _ = model(data)
        scores = y_hat[-1][0, :].detach().cpu().numpy()
    selected = np.where(scores > model_threshold)[0].astype(np.int64)
    return selected, scores


def apply_max_lines(
    selected_ids: np.ndarray,
    scores: Optional[np.ndarray],
    method: str,
    max_lines: Optional[int],
) -> np.ndarray:
    if max_lines is None or max_lines <= 0 or len(selected_ids) <= max_lines:
        return selected_ids

    if method != "input" and scores is not None:
        order = np.argsort(scores[selected_ids])[::-1][:max_lines]
        return np.sort(selected_ids[order])

    sample_positions = np.linspace(0, len(selected_ids) - 1, num=max_lines)
    sample_positions = np.round(sample_positions).astype(np.int64)
    sample_positions = np.clip(sample_positions, 0, len(selected_ids) - 1)
    sample_positions = np.unique(sample_positions)
    return selected_ids[sample_positions]


def error_colormap(mask: np.ndarray) -> np.ndarray:
    colors = np.zeros((mask.shape[0], 4), dtype=np.float32)
    colors[mask] = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    colors[~mask] = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return colors


def evaluate_selection(scene_data: Dict, selected_ids: np.ndarray, geod_threshold: float) -> Dict:
    idx_sort_t = scene_data["idx_sort"].T
    matches = idx_sort_t[selected_ids].astype(np.int64)
    corr0 = scene_data["kp_i"][matches[:, 0]]
    corr1 = scene_data["kp_j"][matches[:, 1]]
    x1_sel = scene_data["x1"][matches[:, 0]]
    x2_sel = scene_data["x2_ordered"][selected_ids]

    errors = get_episym(x1_sel, x2_sel, scene_data["dR"], scene_data["dt"])
    inlier_mask = errors < geod_threshold

    raw_errors = get_episym(
        scene_data["x1"][scene_data["idx_sort"][0].astype(np.int64)],
        scene_data["x2_ordered"],
        scene_data["dR"],
        scene_data["dt"],
    )
    total_gt_inliers = int((raw_errors < geod_threshold).sum())
    inlier_count = int(inlier_mask.sum())
    selected_count = int(len(selected_ids))
    precision = float(inlier_count / selected_count) if selected_count else 0.0
    recall = float(inlier_count / total_gt_inliers) if total_gt_inliers else 0.0

    return {
        "corr0": corr0,
        "corr1": corr1,
        "color": error_colormap(inlier_mask),
        "selected_count": selected_count,
        "inlier_count": inlier_count,
        "precision": precision,
        "recall": recall,
    }


def reorder_to_requested_input(scene_data: Dict, eval_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if scene_data["pair_order_matches_input"]:
        return (
            scene_data["image0"],
            scene_data["image1"],
            eval_data["corr0"],
            eval_data["corr1"],
            eval_data["color"],
        )

    return (
        scene_data["image1"],
        scene_data["image0"],
        eval_data["corr1"],
        eval_data["corr0"],
        eval_data["color"],
    )


def save_visualization(
    output_path: str,
    image_left: np.ndarray,
    image_right: np.ndarray,
    corr_left: np.ndarray,
    corr_right: np.ndarray,
    color: np.ndarray,
    dpi: int,
    panel_size: float,
    line_width: float,
) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(panel_size * 2, panel_size * 0.75), dpi=dpi)
    for idx, img in enumerate([image_left, image_right]):
        ax[idx].imshow(img)
        ax[idx].set_xticks([])
        ax[idx].set_yticks([])
        for spine in ax[idx].spines.values():
            spine.set_visible(False)

    fig.canvas.draw()
    trans_figure = fig.transFigure.inverted()
    fkpts0 = trans_figure.transform(ax[0].transData.transform(corr_left))
    fkpts1 = trans_figure.transform(ax[1].transData.transform(corr_right))
    fig.lines = [
        matplotlib.lines.Line2D(
            (fkpts0[i, 0], fkpts1[i, 0]),
            (fkpts0[i, 1], fkpts1[i, 1]),
            zorder=1,
            transform=fig.transFigure,
            c=color[i],
            linewidth=line_width,
        )
        for i in range(len(corr_left))
    ]

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.01)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def build_default_output_path(scene_data: Dict, method: str, output_format: str) -> str:
    img1_stem = os.path.splitext(os.path.basename(scene_data["requested_img1"]))[0]
    img2_stem = os.path.splitext(os.path.basename(scene_data["requested_img2"]))[0]
    file_name = f"{img1_stem}__{img2_stem}__{method}.{output_format}"
    out_dir = os.path.join(
        SCRIPT_DIR,
        "fig6_outputs",
        scene_data["dataset"],
        scene_data["scene"],
        method,
    )
    ensure_dir(out_dir)
    return os.path.join(out_dir, file_name)


def run_helper_mode(args: argparse.Namespace) -> bool:
    if args.list_scenes:
        list_scenes()
        return True

    if args.list_images:
        if not args.dataset or not args.scene:
            raise ValueError("--list-images requires --dataset and --scene.")
        list_images(args.dataset, args.scene, args.limit)
        return True

    if args.list_pairs:
        if not args.dataset or not args.scene:
            raise ValueError("--list-pairs requires --dataset and --scene.")
        list_pairs(args.dataset, args.scene, args.limit)
        return True

    return False


def main() -> None:
    args = parse_args()

    if run_helper_mode(args):
        return

    if not args.img1 or not args.img2:
        raise ValueError("Main mode requires --img1 and --img2.")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, fallback to CPU.")
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    scene_data = load_scene_pair_from_images(args.img1, args.img2)
    model = build_model(
        method=args.method,
        dataset=scene_data["dataset"],
        device=device,
        checkpoint_override=args.checkpoint,
        module_path_override=args.module_path,
        class_name_override=args.class_name,
        checkpoint_key=args.checkpoint_key,
    )
    selected_ids, scores = select_matches(
        method=args.method,
        model=model,
        scene_data=scene_data,
        device=device,
        model_threshold=args.model_threshold,
    )
    selected_ids = apply_max_lines(
        selected_ids=selected_ids,
        scores=scores,
        method=args.method,
        max_lines=args.max_lines,
    )
    eval_data = evaluate_selection(
        scene_data=scene_data,
        selected_ids=selected_ids,
        geod_threshold=args.geod_threshold,
    )

    output_path = args.output
    if output_path is None:
        output_path = build_default_output_path(scene_data, args.method, args.output_format)
    else:
        output_path = os.path.abspath(output_path)
        ensure_dir(os.path.dirname(output_path))

    image_left, image_right, corr_left, corr_right, color = reorder_to_requested_input(
        scene_data,
        eval_data,
    )
    save_visualization(
        output_path=output_path,
        image_left=image_left,
        image_right=image_right,
        corr_left=corr_left,
        corr_right=corr_right,
        color=color,
        dpi=args.dpi,
        panel_size=args.panel_size,
        line_width=args.line_width,
    )

    print(f"dataset: {scene_data['dataset']}")
    print(f"scene: {scene_data['scene']}")
    print(f"pair: {scene_data['pair_name']}")
    print(f"method: {args.method}")
    print(f"selected: {eval_data['selected_count']}")
    print(f"inliers: {eval_data['inlier_count']}/{eval_data['selected_count']}")
    print(f"precision: {eval_data['precision']:.4f}")
    print(f"recall: {eval_data['recall']:.4f}")
    print(f"output: {output_path}")

    if not scene_data["pair_order_matches_input"]:
        print("note: direct nn-img1-img2.h5 was not found, so the reverse pair file was used internally.")


if __name__ == "__main__":
    main()
