import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocessing.preprocessor import get_transforms
from preprocessing.preprocessor import preprocess_geotiff
import torch
import json
import numpy as np

path = r"D:\Major project\geotiffs\tier1\images"

# hardcoded statistics (compute once earlier)
mean = (78.67216964243309,86.80138104794152,64.97934000711008)
std= (41.37580197971469,36.72895827341534,34.51997208576517)

train_transform = get_transforms(mean, std, train=True)

files = os.listdir(path)

# separate pre and post images
pre_images = [f for f in files if "pre_disaster" in f]
post_images = [f for f in files if "post_disaster" in f]

# sort to keep correct pairing
pre_images.sort()
post_images.sort()

damage_map = {
    "no-damage":0,
    "minor-damage":1,
    "major-damage":2,
    "destroyed":3
}

def load_label(json_path):

    with open(json_path) as f:
        data = json.load(f)

    labels = []

    for building in data["features"]["xy"]:
        damage = building["properties"]["subtype"]

        if damage in damage_map:
            labels.append(damage_map[damage])

    if len(labels) == 0:
        return 0

    #SEVERITY-BASED LABELING (FINAL FIX)
    if 3 in labels:
        return 3   # destroyed
    elif 2 in labels:
        return 2   # major
    elif 1 in labels:
        return 1   # minor
    else:
        return 0   # no damage

label_path = r"D:\Major project\geotiffs\tier1\labels"

pre_list = []
post_list = []
labels = []

for i in range(1000):

    pre_img = preprocess_geotiff(
        os.path.join(path, pre_images[i]),
        mean,
        std,
        train_transform
    )

    post_img = preprocess_geotiff(
        os.path.join(path, post_images[i]),
        mean,
        std,
        train_transform
    )

    # load corresponding label
    json_file = post_images[i].replace(".tif", ".json")

    label = load_label(os.path.join(label_path,json_file))

    pre_list.append(pre_img)
    post_list.append(post_img)
    labels.append(label)

    # print(f"\nSet {i+1}")
    # print("Pre image shape:", pre_img.shape)
    # print("Post image shape:", post_img.shape)
    

pre_tensor = torch.stack(pre_list)
post_tensor = torch.stack(post_list)
labels = torch.tensor(labels, dtype=torch.long)

print("Batch shape:", pre_tensor.shape)
# print("Labels:", labels)



