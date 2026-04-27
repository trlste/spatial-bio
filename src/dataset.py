import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter

class SkinDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, return_index=False):
        self.df           = df.reset_index(drop=True)
        self.image_dir    = image_dir
        self.transform    = transform
        self.return_index = return_index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['isic_id']}.jpg")
        img     = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            img = self.transform(image=img)['image']

        label = row['iddx_processed']
        if self.return_index:
            return img, label, idx
        return img, label

def get_class_weights(df, label_col='iddx_processed', n_classes=5, device='cpu'):
    class_counts = df[label_col].value_counts().sort_index()
    class_counts = class_counts.reindex(range(n_classes), fill_value=1)
    weights      = 1.0 / class_counts.values
    weights      = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32).to(device)


def get_sample_weights(df, label_col='iddx_processed'):
    """Per-sample weights for WeightedRandomSampler.

    Each sample receives weight = 1 / sqrt(class_count) so that minority
    classes are oversampled but not aggressively. Returns a float64
    numpy array of length len(df).
    """
    class_counts = df[label_col].value_counts()
    return df[label_col].map(lambda c: 1.0 / np.sqrt(class_counts[c])).to_numpy(dtype='float64')

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
                 train_transform=None, val_transform=None, device='cuda',
                 return_index=False):
    """Build train/val datasets for one fold.

    return_index: when True, train_dataset.__getitem__ returns
        (img, label, idx). Required when using libauc pAUCLoss.
        val_dataset always returns (img, label).
    """
    df = get_folds(csv_path, n_folds=n_folds, seed=seed)
    label_categories = df['iddx_processed'].astype('category').cat.categories.tolist()
    df['iddx_processed'] = df['iddx_processed'].astype('category').cat.codes

    train_df = df[df['fold'] != fold]
    val_df   = df[df['fold'] == fold]

    train_dataset  = SkinDataset(train_df, image_dir, transform=train_transform,
                                  return_index=return_index)
    val_dataset    = SkinDataset(val_df,   image_dir, transform=val_transform,
                                  return_index=False)
    class_weights  = get_class_weights(train_df, device=device)
    sample_weights = get_sample_weights(train_df)

    return train_dataset, val_dataset, class_weights, label_categories, sample_weights


def apply_smoteenn(model, dataset, device='cuda', batch_size=64):
    """Extract embeddings, then apply light undersampling + SMOTE.

    Returns:
        X_resampled (np.ndarray): resampled feature vectors
        y_resampled (np.ndarray): resampled labels
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                imgs, labels, _ = batch
            else:
                imgs, labels = batch

            imgs = imgs.to(device)
            embs = model.extract_features(imgs)
            all_embeddings.append(embs.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.concatenate(all_embeddings, axis=0)
    y = np.concatenate(all_labels, axis=0).astype(np.int64).ravel()

    print(f"Before - shape: {X.shape}, dist: {np.bincount(y)}")

    class_counts = Counter(y)
    min_count = min(class_counts.values())
    target_count = min_count * 10
    sampling_strategy = {
        cls: min(count, target_count)
        for cls, count in class_counts.items()
    }

    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    X, y = rus.fit_resample(X, y)

    X = np.ascontiguousarray(X)
    y = np.ascontiguousarray(y).astype(np.int64)

    print(f"Before SMOTEENN - X: {X.shape}, class dist: {np.bincount(y)}")

    smote = SMOTE(k_neighbors=3, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_res = np.ascontiguousarray(X_res)
    y_res = np.ascontiguousarray(y_res).astype(np.int64)

    print(f"After  SMOTEENN - X: {X_res.shape}, class dist: {np.bincount(y_res)}")

    return X_res, y_res


def apply_smoteenn_naive(model, dataset, device='cuda', batch_size=256):
    """Extract embeddings, then apply vanilla SMOTEENN."""
    from imblearn.combine import SMOTEENN

    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                imgs, labels, _ = batch
            else:
                imgs, labels = batch

            imgs = imgs.to(device)
            embs = model.extract_features(imgs)
            all_embeddings.append(embs.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.concatenate(all_embeddings, axis=0)
    y = np.concatenate(all_labels, axis=0).astype(np.int64).ravel()

    print(f"Before SMOTEENN - X: {X.shape}, class dist: {np.bincount(y)}")

    smoteenn = SMOTEENN(random_state=42)
    X_res, y_res = smoteenn.fit_resample(X, y)

    print(f"After  SMOTEENN - X: {X_res.shape}, class dist: {np.bincount(y_res)}")

    return X_res, y_res
