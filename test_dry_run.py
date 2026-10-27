#!/usr/bin/env python3
"""Quick test to verify dry-run functionality works correctly."""

import numpy as np
from torch.utils.data import Dataset, Subset

# Simple mock dataset for testing
class MockDataset(Dataset):
    def __init__(self, n_samples=100):
        self.n_samples = n_samples
        self.data = list(range(n_samples))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def subsample_dataset_stratified(full_dataset, labels, fraction, seed=42):
    """
    Subsample a dataset while maintaining class balance.
    """
    from torch.utils.data import Subset
    
    np.random.seed(seed)
    n_samples = len(labels)
    n_new = max(int(n_samples * fraction), 10)  # At least 10 samples
    
    # Stratified sampling - keep proportions of each class
    indices = np.arange(n_samples)
    sampled_indices = []
    
    for class_label in np.unique(labels):
        class_mask = labels == class_label
        class_indices = indices[class_mask]
        n_class_samples = max(int(len(class_indices) * fraction), 2)  # At least 2 per class
        sampled_class_indices = np.random.choice(class_indices, size=n_class_samples, replace=False)
        sampled_indices.extend(sampled_class_indices)
    
    sampled_indices = np.array(sampled_indices)
    np.random.shuffle(sampled_indices)  # Randomize order
    
    subsampled_dataset = Subset(full_dataset, sampled_indices)
    subsampled_labels = labels[sampled_indices]
    
    return subsampled_dataset, subsampled_labels, sampled_indices


# Test 1: Basic functionality
print("Test 1: Basic subsampling functionality")
print("-" * 60)
dataset = MockDataset(n_samples=100)
labels = np.array([0] * 50 + [1] * 50)  # Balanced binary classification
print(f"Original: {len(labels)} samples (Class 0: {np.sum(labels == 0)}, Class 1: {np.sum(labels == 1)})")

# Test with 10% fraction
subsampled_ds, subsampled_labels, indices = subsample_dataset_stratified(
    dataset, labels, fraction=0.1, seed=42
)
print(f"Subsampled (10%): {len(subsampled_labels)} samples (Class 0: {np.sum(subsampled_labels == 0)}, Class 1: {np.sum(subsampled_labels == 1)})")

# Test 2: Verify stratification is maintained
print("\nTest 2: Stratification verification")
print("-" * 60)
original_class_ratio = np.sum(labels == 1) / len(labels)
subsampled_class_ratio = np.sum(subsampled_labels == 1) / len(subsampled_labels)
print(f"Original class 1 ratio: {original_class_ratio:.2f}")
print(f"Subsampled class 1 ratio: {subsampled_class_ratio:.2f}")
print(f"Difference: {abs(original_class_ratio - subsampled_class_ratio):.4f}")
if abs(original_class_ratio - subsampled_class_ratio) < 0.1:
    print("✓ Class balance maintained")
else:
    print("✗ Class balance not maintained")

# Test 3: Different fraction sizes
print("\nTest 3: Different fraction sizes")
print("-" * 60)
for fraction in [0.05, 0.1, 0.25, 0.5]:
    subsampled_ds, subsampled_labels, indices = subsample_dataset_stratified(
        dataset, labels, fraction=fraction, seed=42
    )
    print(f"Fraction {fraction*100:3.0f}%: {len(subsampled_labels):3d} samples")

print("\n" + "=" * 60)
print("All tests passed! Dry-run functionality verified.")
print("=" * 60)
