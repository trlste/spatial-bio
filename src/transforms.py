import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 135
AUG_PROB = 0.75

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),

    # --- Geometry ---
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=45,
        border_mode=0,   # black padding, avoids edge artifacts
        p=AUG_PROB
    ),

    # --- Color/Contrast ---
    # OneOf means only one of these runs per image, avoids over-augmenting color
    A.OneOf([
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
    ], p=AUG_PROB),

    # --- Simulate dermoscopy artifacts ---
    # Mimics hair, bubbles, and other occlusions common in skin images
    A.CoarseDropout(
        max_holes=8,
        max_height=16,
        max_width=16,
        min_holes=1,
        fill_value=0,
        p=0.25
    ),

    # --- Normalize and convert ---
    A.Normalize(
        mean=[0.4815, 0.4578, 0.4082],
        std=[0.2686, 0.2613, 0.2758],
        max_pixel_value=255.0
    ),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(
        mean=[0.4815, 0.4578, 0.4082],
        std=[0.2686, 0.2613, 0.2758],
        max_pixel_value=255.0
    ),
    ToTensorV2()
])