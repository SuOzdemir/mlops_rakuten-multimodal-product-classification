# ============================================================
# dataset.py
# Rakuten Image Dataset – PyTorch Dataset wrapper
#
# Note: return_idx=True is used during feature export so that
# each sample can be matched back to its DataFrame row.
# ============================================================

import torch
from PIL import Image
from torch.utils.data import Dataset


class RakutenImageDataset(Dataset):
    """
    PyTorch Dataset for the Rakuten product image classification task.

    Args:
        df         : DataFrame with at least image path and label columns.
        transform  : torchvision transform pipeline (optional).
        path_col   : Column name that holds the absolute image path string.
        label_col  : Column name that holds the integer class label.
        return_idx : If True, __getitem__ returns (image, label, idx).
                     Required for ordered feature export (shuffle=False).
    """

    def __init__(
        self,
        df,
        transform=None,
        path_col: str = "image_path_local",
        label_col: str = "label_id",
        return_idx: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.path_col = path_col
        self.label_col = label_col
        self.return_idx = return_idx

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, self.path_col]
        label = int(self.df.loc[idx, self.label_col])

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        except Exception as e:
            print(f"[WARNING] Error loading image {img_path}: {e}")
            image = torch.zeros((3, 224, 224))

        if self.return_idx:
            return image, label, idx
        return image, label
