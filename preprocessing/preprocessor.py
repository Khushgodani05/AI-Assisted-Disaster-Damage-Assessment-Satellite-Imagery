import os
import numpy as np
import rasterio
from tqdm import tqdm
import torch
from torchvision import transforms


############################################################
# 1. COMPUTE DATASET MEAN AND STD
############################################################

def compute_dataset_stats(path):

    sum_r, sum_g, sum_b = 0.0, 0.0, 0.0
    sum_sq_r, sum_sq_g, sum_sq_b = 0.0, 0.0, 0.0
    n_pixels = 0

    for fname in tqdm(os.listdir(path)):

        if not fname.endswith(".tif"):
            continue

        with rasterio.open(os.path.join(path, fname)) as src:

            # masked=True ignores NoData values
            r = src.read(1, masked=True).astype(np.float32)
            g = src.read(2, masked=True).astype(np.float32)
            b = src.read(3, masked=True).astype(np.float32)

            r = r.flatten()
            g = g.flatten()
            b = b.flatten()

            sum_r += r.sum()
            sum_g += g.sum()
            sum_b += b.sum()

            sum_sq_r += (r**2).sum()
            sum_sq_g += (g**2).sum()
            sum_sq_b += (b**2).sum()

            n_pixels += len(r)

    mean_r = sum_r / n_pixels
    mean_g = sum_g / n_pixels
    mean_b = sum_b / n_pixels

    std_r = np.sqrt(sum_sq_r / n_pixels - mean_r**2)
    std_g = np.sqrt(sum_sq_g / n_pixels - mean_g**2)
    std_b = np.sqrt(sum_sq_b / n_pixels - mean_b**2)

    mean = (mean_r, mean_g, mean_b)
    std = (std_r, std_g, std_b)

    print("Global Mean (R,G,B):", mean)
    print("Global Std  (R,G,B):", std)

    return mean, std


############################################################
# 2. NORMALIZE SINGLE IMAGE
############################################################

def normalize_image(src, mean, std):

    mean_r, mean_g, mean_b = mean
    std_r, std_g, std_b = std

    r = src.read(1).astype(np.float32)
    g = src.read(2).astype(np.float32)
    b = src.read(3).astype(np.float32)

    # Z-score normalization
    r = (r - mean_r) / std_r
    g = (g - mean_g) / std_g
    b = (b - mean_b) / std_b

    image = np.dstack((r, g, b))  # (H, W, 3)

    return image


############################################################
# 3. LOAD NORMALIZED IMAGES
############################################################

def load_normalized_images(path, mean, std, num_images=4):

    images = []
    files = os.listdir(path)

    for i in range(num_images):

        with rasterio.open(os.path.join(path, files[i])) as src:

            img = normalize_image(src, mean, std)

            images.append(img)

            print(
                "Image shape:",
                img.shape,
                "min:",
                img.min(),
                "max:",
                img.max()
            )

    return images


############################################################
# 4. DATA AUGMENTATION + TRANSFORMS
############################################################

def get_transforms(mean, std, image_size=256, train=True):

    if train:

        transform = transforms.Compose([

            transforms.ToTensor(),

            # Data Augmentation
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),

            # Resize
            transforms.Resize((image_size, image_size)),

            # Normalize
            transforms.Normalize(
                mean=list(mean),
                std=list(std)
            )

        ])

    else:

        # Validation / Test transforms
        transform = transforms.Compose([

            transforms.ToTensor(),

            transforms.Resize((image_size, image_size)),

            transforms.Normalize(
                mean=list(mean),
                std=list(std)
            )

        ])

    return transform


############################################################
# 5. FINAL PREPROCESS FUNCTION
############################################################

def preprocess_image(image, transform=None):

    if transform is not None:

        image = transform(image)

    else:

        image = torch.tensor(image).permute(2,0,1)

    return image


############################################################
# 6. COMPLETE PIPELINE FUNCTION
############################################################

def preprocess_geotiff(path, mean, std, transform=None):

    with rasterio.open(path) as src:

        image = normalize_image(src, mean, std)

    image = preprocess_image(image, transform)

    return image