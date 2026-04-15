import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold

class SkinDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['isic_id']}.jpg")
        img     = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            img = self.transform(image=img)['image']

        label = row['iddx_processed']
        return img, label

def get_class_weights(df, label_col='iddx_processed', n_classes=5, device='cpu'):
    class_counts = df[label_col].value_counts().sort_index()
    class_counts = class_counts.reindex(range(n_classes), fill_value=1)
    weights      = 1.0 / class_counts.values
    weights      = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32).to(device)

def get_folds(csv_path, n_folds=5, seed=42):
    df = pd.read_csv(csv_path)

    df['fold'] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # stratify on label, group on isic_id so the same lesion
    # doesn't appear in both train and val
    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(df, y=df['iddx_processed'], groups=df['isic_id'])
    ):
        df.loc[val_idx, 'fold'] = fold

    return df


def get_datasets(csv_path, image_dir, fold, n_folds=5, seed=42,
                 train_transform=None, val_transform=None, device='cuda'):
    df = get_folds(csv_path, n_folds=n_folds, seed=seed)
    label_categories = df['iddx_processed'].astype('category').cat.categories.tolist()
    df['iddx_processed'] = df['iddx_processed'].astype('category').cat.codes

    train_df = df[df['fold'] != fold]
    val_df   = df[df['fold'] == fold]

    train_dataset = SkinDataset(train_df, image_dir, transform=train_transform)
    val_dataset   = SkinDataset(val_df,   image_dir, transform=val_transform)
    class_weights  = get_class_weights(train_df, device=device)  # computed on train only

    return train_dataset, val_dataset, class_weights, label_categories