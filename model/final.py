if __name__ == "__main__":

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm
    import collections
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    from model.major_project import SiameseMultiFusion
    from preprocessing.preprocess_train import pre_tensor, post_tensor, labels

    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Label distribution:", torch.bincount(labels))

    X_pre_train, X_pre_val, X_post_train, X_post_val, y_train, y_val = train_test_split(
        pre_tensor,
        post_tensor,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    train_dataset = TensorDataset(X_pre_train, X_post_train, y_train)
    val_dataset = TensorDataset(X_pre_val, X_post_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
    )

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
# CLASS WEIGHTS
# -------------------------
class_counts = torch.bincount(labels).float()
weights = class_counts.sum() / class_counts
weights = weights / weights.max()

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

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for pre, post, label in loop:

        # Move to device
        pre = pre.to(device)
        post = post.to(device)
        label = label.to(device)

        emb_pre, emb_post, _ = model(pre, post)

        diff = torch.abs(emb_pre - emb_post)

        preds = classifier(diff*2)

        loss = criterion(preds, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()

    val_loss = 0            
    all_preds = []
    all_labels = []

    with torch.inference_mode():

        for pre, post, label in val_loader:

            pre = pre.to(device)
            post = post.to(device)
            label = label.to(device)

            emb_pre, emb_post, _ = model(pre, post)

            diff = torch.abs(emb_pre - emb_post)
            

            preds = classifier(diff*2)

            loss = criterion(preds, label)
            val_loss += loss.item()

            predicted = torch.argmax(preds, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)

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

    for pre, post, label in val_loader:

        pre = pre.to(device)
        post = post.to(device)
        label = label.to(device)

        emb_pre, emb_post, _ = model(pre, post)

        diff = torch.abs(emb_pre - emb_post)

        preds = classifier(diff*2)

        predicted = torch.argmax(preds, dim=1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

print("\nFINAL RESULTS ON UNSEEN DATA")
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("Precision:", precision_score(all_labels, all_preds, average="macro", zero_division=0))
print("Recall:", recall_score(all_labels, all_preds, average="macro", zero_division=0))
print("F1 Score:", f1_score(all_labels, all_preds, average="macro", zero_division=0))