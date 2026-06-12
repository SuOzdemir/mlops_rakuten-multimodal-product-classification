# ============================================================
# dataset.py
# Rakuten Text Dataset – PyTorch Dataset wrapper for CamemBERT
# ============================================================

import torch
from torch.utils.data import Dataset


class RakutenTextDataset(Dataset):
    """
    PyTorch Dataset for the Rakuten product text classification task.

    Combines 'designation' and 'description' columns into a single
    input string, then tokenizes it for CamemBERT.

    Args:
        df         : DataFrame with 'designation', 'description', and label column.
        tokenizer  : HuggingFace tokenizer (CamemBERT).
        max_length : Maximum token sequence length (default 128).
        label_col  : Column name that holds the integer class label.
    """

    def __init__(
        self,
        df,
        tokenizer,
        max_length: int = 128,
        label_col: str = "label_id",
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_col = label_col

        # Combine designation + description for richer features
        self.texts = (
            self.df["designation"].fillna("") + " " + self.df["description"].fillna("")
        ).values
        self.labels = self.df[label_col].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> dict:
        text  = str(self.texts[idx])
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels":         label,
        }
