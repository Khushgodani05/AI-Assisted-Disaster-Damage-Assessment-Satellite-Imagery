import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import collections

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from major_project import SiameseMultiFusion
from preprocessing.preprocess_train import pre_tensor, post_tensor, labels

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Check Label Distribution
# -------------------------
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

# -------------------------
# Model
# -------------------------
model = SiameseMultiFusion().to(device)
classifier = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 4)
).to(device)

# -------------------------
# CLASS WEIGHTS (MAIN FIX)
# -------------------------
class_counts = torch.bincount(labels) + 1e-6
weights = 1.0 / class_counts.float()
weights = weights / weights.min()

print("Class weights:", weights)

criterion = nn.CrossEntropyLoss(weight=weights.to(device))

# -------------------------
# Optimizer
# -------------------------
optimizer = optim.Adam(
    list(model.parameters()) + list(classifier.parameters()),
    lr=0.001,
)

# -------------------------
# Training Config
# -------------------------
epochs = 25
patience = 5
best_val_loss = float("inf")
counter = 0

# -------------------------
# Training Loop
# -------------------------
for epoch in range(epochs):

    model.train()
    total_loss = 0

    # Shuffle indices
    indices = torch.randperm(len(X_pre_train))

    loop = tqdm(indices, desc=f"Epoch {epoch+1}/{epochs}")

    for i in loop:

        pre = X_pre_train[i].unsqueeze(0)
        post = X_post_train[i].unsqueeze(0)
        label = y_train[i].unsqueeze(0)

        emb_pre, emb_post, _ = model(pre, post)

        diff = torch.abs(emb_pre - emb_post)

        # Important scaling
        diff = diff * 2

        preds = classifier(diff)

        loss = criterion(preds, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(X_pre_train)

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()

    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.inference_mode():

        for i in range(len(X_pre_val)):

            pre = X_pre_val[i].unsqueeze(0)
            post = X_post_val[i].unsqueeze(0)
            label = y_val[i].unsqueeze(0)

            emb_pre, emb_post, _ = model(pre, post)

            diff = torch.abs(emb_pre - emb_post)
            diff = diff * 2

            preds = classifier(diff)

            loss = criterion(preds, label)
            val_loss += loss.item()

            predicted = torch.argmax(preds, dim=1)

            all_preds.append(predicted.item())
            all_labels.append(label.item())

    avg_val_loss = val_loss / len(X_pre_val)

    # -------------------------
    # Metrics
    # -------------------------
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("Prediction distribution:", collections.Counter(all_preds))
    print("-" * 50)

    # -------------------------
    # CHECKPOINT
    # -------------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0

        torch.save({
            "model": model.state_dict(),
            "classifier": classifier.state_dict()
        }, "best_model.pth")

        print("✅ Best model saved!")

    else:
        counter += 1
        print(f"⚠️ No improvement {counter}/{patience}")

    # -------------------------
    # EARLY STOPPING
    # -------------------------
    if counter >= patience:
        print("🛑 Early stopping triggered")
        break


# -------------------------
# FINAL EVALUATION
# -------------------------
checkpoint = torch.load("best_model.pth")

model.load_state_dict(checkpoint["model"])
classifier.load_state_dict(checkpoint["classifier"])

model.eval()

all_preds = []
all_labels = []

with torch.inference_mode():

    for i in range(len(X_pre_val)):

        pre = X_pre_val[i].unsqueeze(0)
        post = X_post_val[i].unsqueeze(0)
        label = y_val[i].unsqueeze(0)

        emb_pre, emb_post, _ = model(pre, post)

        diff = torch.abs(emb_pre - emb_post)
        diff = diff * 2

        preds = classifier(diff)

        predicted = torch.argmax(preds, dim=1)

        all_preds.append(predicted.item())
        all_labels.append(label.item())

print("\nFINAL RESULTS ON UNSEEN DATA")
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("Precision:", precision_score(all_labels, all_preds, average="macro", zero_division=0))
print("Recall:", recall_score(all_labels, all_preds, average="macro", zero_division=0))
print("F1 Score:", f1_score(all_labels, all_preds, average="macro", zero_division=0))