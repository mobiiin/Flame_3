import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from data_loading import UnifiedFlameDataset, get_default_transform


class LightweightCNN(nn.Module):
    def __init__(self, in_channels, base_filters=16, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(base_filters * 4, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        logits = self.classifier(self.dropout(x))
        embeddings = F.normalize(x, dim=1)
        return logits, embeddings


def prepare_model(modality, base_filters=16, dropout=0.2):
    in_channels = 3 if modality == "RGB" else 1 if modality in ["TIFF", "THERMAL"] else 4
    return LightweightCNN(in_channels=in_channels, base_filters=base_filters, dropout=dropout)


def prepare_inputs(modality, rgb_imgs, therm_imgs):
    if modality == "RGB":
        return rgb_imgs
    therm_single = therm_imgs.mean(dim=1, keepdim=True) if therm_imgs.shape[1] != 1 else therm_imgs
    if modality in ["TIFF", "THERMAL"]:
        return therm_single
    return torch.cat([rgb_imgs, therm_single], dim=1)


def subset_first_ten(dataset):
    return Subset(dataset, range(min(10, len(dataset))))


def create_one_batch_loader(dataset):
    if len(dataset) == 0:
        raise RuntimeError("Dataset has 0 samples after subsetting.")
    batch_size = min(4, len(dataset))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def run_sanity_check(datasets, modality, thermal_source=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_default_transform(image_size=224)

    train_ds = UnifiedFlameDataset(
        mode=modality,
        datasets=datasets,
        transform=transform,
        thermal_source=thermal_source,
        is_test_set=False,
    )
    val_ds = UnifiedFlameDataset(
        mode=modality,
        datasets=datasets,
        transform=transform,
        thermal_source=thermal_source,
        is_test_set=False,
    )
    test_ds = UnifiedFlameDataset(
        mode=modality,
        datasets=["TEST_SET"],
        transform=transform,
        thermal_source=thermal_source,
        is_test_set=True,
    )

    train_ds = subset_first_ten(train_ds)
    val_ds = subset_first_ten(val_ds)
    test_ds = subset_first_ten(test_ds)
    print("[SUCCESS] Datasets loaded.")

    train_loader = create_one_batch_loader(train_ds)
    val_loader = create_one_batch_loader(val_ds)
    test_loader = create_one_batch_loader(test_ds)

    model = prepare_model(modality).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for _ in range(1):
        model.train()
        for rgb_imgs, therm_imgs, labels in train_loader:
            rgb_imgs = rgb_imgs.to(device)
            therm_imgs = therm_imgs.to(device)
            labels = labels.to(device)
            inputs = prepare_inputs(modality, rgb_imgs, therm_imgs)
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            break

        model.eval()
        with torch.no_grad():
            for rgb_imgs, therm_imgs, labels in val_loader:
                rgb_imgs = rgb_imgs.to(device)
                therm_imgs = therm_imgs.to(device)
                labels = labels.to(device)
                inputs = prepare_inputs(modality, rgb_imgs, therm_imgs)
                logits, _ = model(inputs)
                _ = criterion(logits, labels)
                break

            for rgb_imgs, therm_imgs, labels in test_loader:
                rgb_imgs = rgb_imgs.to(device)
                therm_imgs = therm_imgs.to(device)
                labels = labels.to(device)
                inputs = prepare_inputs(modality, rgb_imgs, therm_imgs)
                logits, _ = model(inputs)
                _ = criterion(logits, labels)
                break

    print("[SUCCESS] Model forward and backward pass successful.")


if __name__ == "__main__":
    test_configs = [
        ("flame1__rgb", ["FLAME1"], "RGB", None),
        ("flame2__rgb", ["FLAME2"], "RGB", None),
        ("flame2__thermal_jpg", ["FLAME2"], "THERMAL", "jpg"),
        ("flame2__rgbt_jpg", ["FLAME2"], "RGBT", "jpg"),
        ("flame3__rgb", ["FLAME3"], "RGB", None),
        ("flame3__thermal_jpg", ["FLAME3"], "THERMAL", "jpg"),
        ("flame3__rgbt_jpg", ["FLAME3"], "RGBT", "jpg"),
        ("flame3__thermal_tiff", ["FLAME3"], "THERMAL", "tiff"),
        ("flame3__rgbt_tiff", ["FLAME3"], "RGBT", "tiff"),
        ("flame1_flame2__rgb", ["FLAME1", "FLAME2"], "RGB", None),
        ("flame1_flame3__rgb", ["FLAME1", "FLAME3"], "RGB", None),
        ("flame2_flame3__rgb", ["FLAME2", "FLAME3"], "RGB", None),
        ("flame2_flame3__thermal_jpg", ["FLAME2", "FLAME3"], "THERMAL", "jpg"),
        ("flame2_flame3__rgbt_jpg", ["FLAME2", "FLAME3"], "RGBT", "jpg"),
        ("flame1_flame2_flame3__rgb", ["FLAME1", "FLAME2", "FLAME3"], "RGB", None),
    ]

    for run_name, datasets, modality, thermal_source in test_configs:
        print(f"\n=== Testing Configuration | Modality: {modality} | Datasets: {datasets} ===")
        print(f"Run: {run_name} | Thermal source: {thermal_source or 'n/a'}")
        try:
            run_sanity_check(datasets, modality, thermal_source=thermal_source)
        except Exception:
            traceback.print_exc()
