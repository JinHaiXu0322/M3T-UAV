import json
import os.path as osp
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.dataset import get_resized_wh, get_divisible_wh


class M3TDataset(Dataset):
    # NEW: M3T loader for vis->map / ir->map
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        modality: str = "vis",
        img_resize: int = 640,
        df: int = 8,
        augment_cfg: Optional[Dict[str, Any]] = None,
        enable_augmentation: bool = False,
    ) -> None:
        super().__init__()
        assert split in ["train", "test"]
        assert modality in ["vis", "ir"]
        self.root_dir = root_dir
        self.split = split
        self.modality = modality
        self.img_resize = img_resize
        self.df = df
        self.enable_augmentation = bool(enable_augmentation and split == "train")
        self.augment_cfg = augment_cfg or {}

        split_dir = osp.join(root_dir, f"{split}_dataset")
        json_path = osp.join(split_dir, f"{modality}.json")
        with open(json_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        self.samples: List[Dict[str, Any]] = []
        uav_folder = "uav_vis" if modality == "vis" else "uav_ir"
        for scene_id, scene_info in ann.items():
            uav_list = scene_info["uav_img_path"]
            map_list = scene_info["map_path"]
            pose_list = scene_info.get("pose", [None] * len(uav_list))
            H_list = scene_info["H_matrix"]
            n = min(len(uav_list), len(map_list), len(H_list))
            for i in range(n):
                self.samples.append(
                    {
                        "scene_id": scene_id,
                        "uav_path": osp.join(split_dir, scene_id, uav_folder, uav_list[i]),
                        "map_path": osp.join(split_dir, scene_id, "map", map_list[i]),
                        "H_gt": np.array(H_list[i], dtype=np.float32).reshape(3, 3),
                        "pose": pose_list[i],
                        "pair_name": (uav_list[i], map_list[i]),
                    }
                )

    # NEW: fixed-order stochastic augmentation
    def _build_centered_transform(self, w: int, h: int) -> np.ndarray:
        cx = 0.5 * (w - 1)
        cy = 0.5 * (h - 1)
        T_to = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
        T_back = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)
        return T_to, T_back

    def _sample_uav_geo_transform(self, w: int, h: int) -> np.ndarray:
        # Order must be: Rotation -> Affine -> Translation
        p_rot = float(self.augment_cfg.get("ROTATION_P", 0.4))
        p_aff = float(self.augment_cfg.get("AFFINE_P", 0.4))
        p_trans = float(self.augment_cfg.get("TRANSLATION_P", 0.4))

        max_rot_deg = float(self.augment_cfg.get("MAX_ROT_DEG", 20.0))
        affine_scale_min = float(self.augment_cfg.get("AFFINE_SCALE_MIN", 0.9))
        affine_scale_max = float(self.augment_cfg.get("AFFINE_SCALE_MAX", 1.1))
        max_shear_deg = float(self.augment_cfg.get("MAX_SHEAR_DEG", 8.0))
        max_trans_ratio = float(self.augment_cfg.get("MAX_TRANS_RATIO", 0.08))

        T_to, T_back = self._build_centered_transform(w, h)
        A_total = np.eye(3, dtype=np.float32)

        if np.random.rand() < p_rot:
            angle = np.random.uniform(-max_rot_deg, max_rot_deg)
            a = np.deg2rad(angle)
            R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], dtype=np.float32)
            A_total = (T_back @ R @ T_to) @ A_total

        if np.random.rand() < p_aff:
            scale = np.random.uniform(affine_scale_min, affine_scale_max)
            shear_deg = np.random.uniform(-max_shear_deg, max_shear_deg)
            shear = np.tan(np.deg2rad(shear_deg))
            A = np.array([[scale, shear, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
            A_total = (T_back @ A @ T_to) @ A_total

        if np.random.rand() < p_trans:
            tx = np.random.uniform(-max_trans_ratio * w, max_trans_ratio * w)
            ty = np.random.uniform(-max_trans_ratio * h, max_trans_ratio * h)
            T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
            A_total = T @ A_total

        return A_total

    def _apply_illumination(self, image: np.ndarray, is_map: bool) -> np.ndarray:
        # Illumination is optional and random for UAV and map.
        p_illum = float(self.augment_cfg.get("ILLUMINATION_P", 0.5))
        p_noise = float(self.augment_cfg.get("NOISE_P", 0.3))
        p_fog = float(self.augment_cfg.get("FOG_P", 0.2))
        if np.random.rand() >= p_illum:
            return image

        # map gets lighter perturbation
        b_jit = float(self.augment_cfg.get("MAP_BRIGHTNESS_JITTER", 0.06 if is_map else 0.2))
        c_jit = float(self.augment_cfg.get("MAP_CONTRAST_JITTER", 0.06 if is_map else 0.2))
        max_noise = float(self.augment_cfg.get("MAP_NOISE_STD", 2.0 if is_map else 6.0))

        out = image.astype(np.float32)
        alpha = 1.0 + np.random.uniform(-c_jit, c_jit)  # contrast
        beta = 255.0 * np.random.uniform(-b_jit, b_jit)  # brightness
        out = out * alpha + beta

        if np.random.rand() < p_noise:
            noise = np.random.normal(0.0, max_noise, size=out.shape).astype(np.float32)
            out = out + noise

        if (not is_map) and (np.random.rand() < p_fog):
            fog_strength = np.random.uniform(0.05, 0.18)
            out = out * (1.0 - fog_strength) + 255.0 * fog_strength

        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    def _apply_uav_aug_and_update_h(self, image0: np.ndarray, H_gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image0.shape[:2]
        A = self._sample_uav_geo_transform(w, h)  # old uav -> new uav
        # Warp UAV image with geometric transform.
        image0_aug = cv2.warpPerspective(
            image0, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        # map = H_gt * uav_old, uav_old = A^{-1} * uav_new => map = (H_gt * A^{-1}) * uav_new
        H_new = H_gt @ np.linalg.inv(A)
        return image0_aug, H_new.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_gray_resize(path: str, img_resize: int, df: int):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        h, w = image.shape[:2]
        w_new, h_new = get_resized_wh(w, h, img_resize)
        w_new, h_new = get_divisible_wh(w_new, h_new, df)
        image = cv2.resize(image, (w_new, h_new))
        scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float32)  # [2]
        image = torch.from_numpy(image).float()[None] / 255.0  # [1, h, w]
        return image, scale

    @staticmethod
    def _rescale_homography(H: np.ndarray, scale0: torch.Tensor, scale1: torch.Tensor) -> torch.Tensor:
        # NEW: adjust H from original pixel frame to resized frame.
        s0x, s0y = float(scale0[0]), float(scale0[1])  # orig/new for image0
        s1x, s1y = float(scale1[0]), float(scale1[1])  # orig/new for image1
        S0 = np.array([[s0x, 0.0, 0.0], [0.0, s0y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        S1_inv = np.array([[1.0 / s1x, 0.0, 0.0], [0.0, 1.0 / s1y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        H_new = S1_inv @ H @ S0
        return torch.from_numpy(H_new).float()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        image0_np = cv2.imread(item["uav_path"], cv2.IMREAD_GRAYSCALE)
        image1_np = cv2.imread(item["map_path"], cv2.IMREAD_GRAYSCALE)
        if image0_np is None:
            raise FileNotFoundError(item["uav_path"])
        if image1_np is None:
            raise FileNotFoundError(item["map_path"])

        H_src = item["H_gt"].copy().astype(np.float32)
        if self.enable_augmentation:
            # Geometric transforms on UAV only + GT sync.
            image0_np, H_src = self._apply_uav_aug_and_update_h(image0_np, H_src)
            # Illumination on UAV and optional light illumination on map.
            image0_np = self._apply_illumination(image0_np, is_map=False)
            image1_np = self._apply_illumination(image1_np, is_map=True)

        # Resize after augmentation and rescale homography accordingly.
        h0, w0 = image0_np.shape[:2]
        h1, w1 = image1_np.shape[:2]
        w0_new, h0_new = get_resized_wh(w0, h0, self.img_resize)
        w0_new, h0_new = get_divisible_wh(w0_new, h0_new, self.df)
        w1_new, h1_new = get_resized_wh(w1, h1, self.img_resize)
        w1_new, h1_new = get_divisible_wh(w1_new, h1_new, self.df)
        image0_np = cv2.resize(image0_np, (w0_new, h0_new))
        image1_np = cv2.resize(image1_np, (w1_new, h1_new))
        scale0 = torch.tensor([w0 / w0_new, h0 / h0_new], dtype=torch.float32)  # [2]
        scale1 = torch.tensor([w1 / w1_new, h1 / h1_new], dtype=torch.float32)  # [2]
        H_gt = self._rescale_homography(H_src, scale0, scale1)  # [3, 3]
        image0 = torch.from_numpy(image0_np).float()[None] / 255.0  # [1, h0_new, w0_new]
        image1 = torch.from_numpy(image1_np).float()[None] / 255.0  # [1, h1_new, w1_new]

        data = {
            "image0": image0,
            "image1": image1,
            "depth0": torch.tensor([]),
            "depth1": torch.tensor([]),
            "scale0": scale0,
            "scale1": scale1,
            "H_gt": H_gt,  # [3, 3]
            "gt": {"H_gt": H_gt},  # NEW: explicit gt field requested
            "pose0": torch.tensor(item["pose"], dtype=torch.float32) if item["pose"] is not None else torch.zeros(3),
            "dataset_name": "M3T",
            "scene_id": item["scene_id"],
            "pair_id": idx,
            "pair_names": item["pair_name"],
            "modality": self.modality,
        }
        return data
