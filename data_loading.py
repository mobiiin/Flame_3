import os
import re
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


BASE_DATA_DIR = "/home/mhabibp/Downloads"


def _p(*parts):
    return os.path.join(BASE_DATA_DIR, *parts)


DATASET_PATHS = {
    "FLAME1": {
        "rgb": [
            _p("Flame 1", "Training", "Fire"),
            _p("Flame 1", "Training", "No_Fire"),
            _p("Flame 1", "Test", "Fire"),
            _p("Flame 1", "Test", "No_Fire"),
        ],
    },
    "FLAME2": {
        "rgb": [
            _p("Flame 2", "Original Sized RGB Images", "Fire"),
            _p("Flame 2", "Original Sized RGB Images", "No Fire"),
        ],
        "thermal_jpg": [
            _p("Flame 2", "Original Sized Thermal Images", "Fire"),
            _p("Flame 2", "Original Sized Thermal Images", "No Fire"),
        ],
    },
    "FLAME3": {
        "rgb": [
            _p("Flame 3", "Shoetank", "Images", "Fire", "RGB", "Corrected FOV"),
            _p("Flame 3", "Shoetank", "Images", "No Fire", "RGB", "Corrected FOV"),
            # _p("Flame 3", "Sycan", "Sycan 2A", "Images", "Fire", "RGB", "Corrected FOV"),
            # _p("Flame 3", "Sycan", "Sycan 2A", "Images", "No Fire", "RGB", "Corrected FOV"),
            # _p("Flame 3", "Wilamette", "Rathbone Double Unit", "Images", "Fire", "RGB", "Corrected FOV"),
            # _p("Flame 3", "Wilamette", "Rathbone Double Unit", "Images", "No Fire", "RGB", "Corrected FOV"),
        ],
        "thermal_jpg": [
            _p("Flame 3", "Shoetank", "Images", "Fire", "Thermal", "JPG"),
            _p("Flame 3", "Shoetank", "Images", "No Fire", "Thermal", "JPG"),
            # _p("Flame 3", "Sycan", "Sycan 2A", "Images", "Fire", "Thermal", "JPG"),
            # _p("Flame 3", "Sycan", "Sycan 2A", "Images", "No Fire", "Thermal", "JPG"),
            # _p("Flame 3", "Wilamette", "Rathbone Double Unit", "Images", "Fire", "Thermal", "JPG"),
            # _p("Flame 3", "Wilamette", "Rathbone Double Unit", "Images", "No Fire", "Thermal", "JPG"),
        ],
        "thermal_tiff": [
            _p("Flame 3", "Shoetank", "Images", "Fire", "Thermal", "Celsius TIFF"),
            _p("Flame 3", "Shoetank", "Images", "No Fire", "Thermal", "Celsius TIFF"),
            # _p("Flame 3", "Sycan", "Sycan 2A", "Images", "Fire", "Thermal", "Celsius TIFF"),
            # _p("Flame 3", "Sycan", "Sycan 2A", "Images", "No Fire", "Thermal", "Celsius TIFF"),
            # _p("Flame 3", "Wilamette", "Rathbone Double Unit", "Images", "Fire", "Thermal", "Celsius TIFF"),
            # _p("Flame 3", "Wilamette", "Rathbone Double Unit", "Images", "No Fire", "Thermal", "Celsius TIFF"),
        ],
    },
    "TEST_SET": {
        "rgb": [
            _p("Flame 3", "Sycan", "Sycan 2D", "Images", "Fire", "RGB", "Corrected FOV"),
            _p("Flame 3", "Sycan", "Sycan 2D", "Images", "No Fire", "RGB", "Corrected FOV"),
            _p("Flame 3", "Wilamette", "Hay Pasture", "Images", "Fire", "RGB", "Corrected FOV"),
            _p("Flame 3", "Wilamette", "Office Unit Pre-Burn", "Images", "No Fire", "RGB", "Corrected FOV"),
        ],
        "thermal_jpg": [
            _p("Flame 3", "Sycan", "Sycan 2D", "Images", "Fire", "Thermal", "JPG"),
            _p("Flame 3", "Sycan", "Sycan 2D", "Images", "No Fire", "Thermal", "JPG"),
            _p("Flame 3", "Wilamette", "Hay Pasture", "Images", "Fire", "Thermal", "JPG"),
            _p("Flame 3", "Wilamette", "Office Unit Pre-Burn", "Images", "No Fire", "Thermal", "JPG"),
        ],
        "thermal_tiff": [
            _p("Flame 3", "Sycan", "Sycan 2D", "Images", "Fire", "Thermal", "Celsius TIFF"),
            _p("Flame 3", "Sycan", "Sycan 2D", "Images", "No Fire", "Thermal", "Celsius TIFF"),
            _p("Flame 3", "Wilamette", "Hay Pasture", "Images", "Fire", "Thermal", "Celsius TIFF"),
            _p("Flame 3", "Wilamette", "Office Unit Pre-Burn", "Images", "No Fire", "Thermal", "Celsius TIFF"),
        ],
    },
}


class UnifiedFlameDataset(Dataset):
    TEST_DIR_LIMITS = {
        "Wilamette/Hay Pasture/Images/Fire": 99,
        "Sycan/Sycan 2D/Images/Fire": 116,
    }

    def __init__(self, mode, datasets, transform=None, dataset_paths=None, thermal_source=None, is_test_set=False):
        self.mode, self.thermal_source = self._normalize_mode(mode, thermal_source)
        self.transform = transform
        self.is_test_set = is_test_set
        self.data = []
        self.dataset_paths = dataset_paths or DATASET_PATHS
        self.dataset_counts = {}

        datasets_to_load = ["TEST_SET"] if self.is_test_set else [ds for ds in datasets if ds != "TEST_SET"]
        for ds in datasets_to_load:
            if ds in self.dataset_paths:
                self._load_dataset_files(ds, self.dataset_paths[ds])

        print(
            f"Loaded {len(self.data)} samples for mode {self.mode} "
            f"(thermal source: {self.thermal_source or 'n/a'}, is_test_set={self.is_test_set})"
        )
        for dataset_name, count in self.dataset_counts.items():
            print(f"  {dataset_name}: {count} samples")

    def _normalize_mode(self, mode, thermal_source):
        normalized_mode = mode.upper()
        normalized_thermal_source = thermal_source.lower() if thermal_source else None

        if normalized_mode == "RGB":
            return normalized_mode, None
        if normalized_mode == "THERMAL":
            return normalized_mode, normalized_thermal_source or "jpg"
        if normalized_mode == "TIFF":
            return "THERMAL", "tiff"
        if normalized_mode == "RGBT":
            return normalized_mode, normalized_thermal_source or "jpg"

        raise ValueError(f"Unsupported mode '{mode}'. Use RGB, THERMAL, TIFF, or RGBT.")

    def _label_from_path(self, path):
        path_parts = [
            part.replace("_", " ").strip().lower()
            for part in os.path.normpath(path).split(os.sep)
        ]

        has_no_fire = any(part in {"no fire", "nofire"} for part in path_parts)
        has_fire = any(part == "fire" for part in path_parts)

        if has_no_fire:
            return 0
        if has_fire:
            return 1

        normalized_path = path.replace("_", " ").lower()
        if "no fire" in normalized_path:
            return 0
        if "fire" in normalized_path:
            return 1
        raise ValueError(f"Could not infer label from directory path: {path}")

    def _extract_frame_id(self, stem):
        paren_match = re.search(r"\((\d+)\)", stem)
        if paren_match:
            return paren_match.group(1)

        trailing_match = re.search(r"(\d+)$", stem)
        if trailing_match:
            return trailing_match.group(1)

        return None

    def _build_file_maps(self, directory):
        if not directory or not os.path.isdir(directory):
            return {}, {}

        file_map = {}
        frame_map = {}
        for file_name in sorted(os.listdir(directory)):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".tif")):
                stem = os.path.splitext(file_name)[0].lower()
                file_path = os.path.join(directory, file_name)
                file_map[stem] = file_path

                frame_id = self._extract_frame_id(stem)
                if frame_id and frame_id not in frame_map:
                    frame_map[frame_id] = file_path

        return file_map, frame_map

    def _load_dataset_files(self, dataset_name, ds_paths):
        rgb_dirs = ds_paths.get("rgb", [])
        thermal_dirs = []
        if self.mode != "RGB":
            thermal_key = f"thermal_{self.thermal_source}"
            thermal_dirs = ds_paths.get(thermal_key, [])
            if not thermal_dirs:
                raise ValueError(
                    f"Dataset '{dataset_name}' does not provide {thermal_key} inputs required for mode {self.mode}."
                )

        loaded_before = len(self.data)
        for idx, r_dir in enumerate(rgb_dirs):
            if not os.path.isdir(r_dir):
                continue

            label = self._label_from_path(r_dir)
            thermal_file_map = {}
            thermal_frame_map = {}
            dir_samples = []
            if self.mode != "RGB":
                if idx >= len(thermal_dirs):
                    raise ValueError(
                        f"Dataset '{dataset_name}' has mismatched RGB and thermal directory counts for {self.thermal_source}."
                    )
                thermal_file_map, thermal_frame_map = self._build_file_maps(thermal_dirs[idx])

            for img_name in sorted(os.listdir(r_dir)):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".tif")):
                    r_path = os.path.join(r_dir, img_name)
                    t_path = None

                    if self.mode != "RGB":
                        stem = os.path.splitext(img_name)[0].lower()
                        t_path = thermal_file_map.get(stem)
                        if t_path is None:
                            frame_id = self._extract_frame_id(stem)
                            if frame_id:
                                t_path = thermal_frame_map.get(frame_id)

                    if self.mode == "RGB" or t_path:
                        dir_samples.append((r_path, t_path, label))

            if self.is_test_set:
                for path_substring, limit in self.TEST_DIR_LIMITS.items():
                    if path_substring in r_dir:
                        print(f"Limiting {r_dir} to {limit} samples.")
                        random.shuffle(dir_samples)
                        dir_samples = dir_samples[: min(limit, len(dir_samples))]
                        break

            self.data.extend(dir_samples)

        self.dataset_counts[dataset_name] = len(self.data) - loaded_before

    def __len__(self):
        return len(self.data)

    def _prepare_thermal_image(self, therm_path):
        t_img = Image.open(therm_path)
        if self.thermal_source == "tiff":
            t_arr = np.array(t_img).astype(np.float32)
            t_arr = 255 * (t_arr - t_arr.min()) / (t_arr.max() - t_arr.min() + 1e-6)
            return Image.fromarray(t_arr).convert("L")
        return t_img.convert("RGB")

    def __getitem__(self, index):
        rgb_path, therm_path, label = self.data[index]

        rgb_img = Image.open(rgb_path).convert("RGB")
        if self.transform:
            rgb_img = self.transform(rgb_img)

        therm_img = torch.zeros_like(rgb_img)
        if therm_path:
            t_img = self._prepare_thermal_image(therm_path)
            if self.transform:
                therm_img = self.transform(t_img)

        return rgb_img, therm_img, label


def get_default_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
