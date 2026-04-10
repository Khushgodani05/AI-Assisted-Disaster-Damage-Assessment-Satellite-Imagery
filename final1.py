import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import collections

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from major_project import SiameseMultiFusion
from preprocessing.preprocess_train import pre_tensor, post_tensor, labels

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Label distribution:", torch.bincount(labels))

# -------------------------
# Train-Val Split (STRATIFIED)
# -------------------------
X_pre_train, X_pre_val, X_post_train, X_post_val, y_train, y_val = train_test_split(
    pre_tensor,
    post_tensor,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels   
)

# Move to device
X_pre_train = X_pre_train.to(device)
X_post_train = X_post_train.to(device)
y_train = y_train.to(device)

X_pre_val = X_pre_val.to(device)
X_post_val = X_post_val.to(device)
y_val = y_val.to(device)

from major_project import (
    SiameseMultiFusion,
    SimpleCNN,
    SiameseCNN,
    TransformerOnly
)

models_dict = {
    "SimpleCNN": SimpleCNN(),
    "SiameseCNN": SiameseCNN(),
    "TransformerOnly": TransformerOnly(),
    "ProposedModel": SiameseMultiFusion()
}

results = {}

class_counts = torch.bincount(labels) + 1e-6
weights = 1.0 / class_counts.float()
weights = weights / weights.min()

print("Class weights:", weights)

for model_name, model in models_dict.items():

    print(f"\n🚀 Training {model_name}")
    model = model.to(device)

    # Classifier only for proposed model
    if model_name == "ProposedModel":
        classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        ).to(device)
        params = list(model.parameters()) + list(classifier.parameters())
    else:
        classifier = None
        params = model.parameters()

    optimizer = optim.Adam(params, lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    # -------- TRAIN --------
    for epoch in range(10):   # keep smaller for comparison
        model.train()
        total_loss = 0

        for i in range(len(X_pre_train)):
            pre = X_pre_train[i].unsqueeze(0)
            post = X_post_train[i].unsqueeze(0)
            label = y_train[i].unsqueeze(0)

            if model_name == "ProposedModel":
                emb_pre, emb_post, _ = model(pre, post)
                diff = torch.abs(emb_pre - emb_post)
                preds = classifier(diff)
            else:
                preds = model(pre, post)

            loss = criterion(preds, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    # -------- EVALUATION --------
    model.eval()
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for i in range(len(X_pre_val)):
            pre = X_pre_val[i].unsqueeze(0)
            post = X_post_val[i].unsqueeze(0)
            label = y_val[i].unsqueeze(0)

            if model_name == "ProposedModel":
                emb_pre, emb_post, _ = model(pre, post)
                diff = torch.abs(emb_pre - emb_post)
                preds = classifier(diff)
            else:
                preds = model(pre, post)

            predicted = torch.argmax(preds, dim=1)

            all_preds.append(predicted.item())
            all_labels.append(label.item())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    results[model_name] = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

# -------- FINAL TABLE --------
print("\n📊 FINAL COMPARISON")
for model, metrics in results.items():
    print(f"\n{model}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")