import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# --- GPU info ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dostępne GPU:", torch.cuda.device_count())
print("Używane urządzenie:", device)

# --- Parametry ---
IMG_SIZE = 512
BATCH_SIZE = 32
N_EPOCHS = 5
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pth")

# --- Dane ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root='walnuts/train', transform=transform)
val_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = datasets.ImageFolder(root='walnuts/test_public', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
class_names = test_dataset.classes
print("Klasy testowe:", class_names)

# --- Model ---
class AnomalyDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # --- ENCODER ---
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 512 -> 256
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 256 -> 128
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 128 -> 64
            nn.ReLU(),


        )

        # Bottleneck
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(128 * 64 * 64, 1024)
        self.fc_dec = nn.Linear(1024, 128 * 64 * 64)

        # --- DECODER ---
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),   # 64 -> 128
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),    # 128 -> 256
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),     # 256 -> 512
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = torch.relu(self.fc_enc(x))

        x = torch.relu(self.fc_dec(x))
        x = x.view(-1, 128, 64, 64)

        x = self.decoder(x)
        return x



model = AnomalyDetector().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- Wczytanie lub trenowanie (z opcją douczania) ---
if os.path.exists(MODEL_PATH):
    print(f"Wczytywanie istniejącego modelu z: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    if N_EPOCHS > 0:
        print(f"Douczanie modelu przez {N_EPOCHS} epok...")
        for epoch in range(N_EPOCHS):
            model.train()
            train_loss = 0.0
            for imgs, _ in train_loader:
                imgs = imgs.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, imgs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print(f"[Douczanie] Epoch [{epoch + 1}/{N_EPOCHS}] Loss: {train_loss / len(train_loader):.4f}")

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model został douczony i zapisany: {MODEL_PATH}")
    else:
        print("Pomijam trenowanie (N_EPOCHS = 0).")

else:
    print("Trenowanie modelu od zera..")
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"[Nowy model] Epoch [{epoch + 1}/{N_EPOCHS}] Loss: {train_loss / len(train_loader):.8f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Nowy model zapisano w: {MODEL_PATH}")

# --- TEST NA CAŁYM ZBIORZE ---
model.eval()
correct = 0
total = 0
results = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        imgs_np = imgs[0].cpu().numpy().transpose(1,2,0)
        outputs_np = outputs[0].cpu().numpy().transpose(1,2,0)
        anomaly_map = np.mean(np.abs(imgs_np - outputs_np), axis=-1)
        anomaly_score = np.max(anomaly_map)

        # Klasyfikacja
        predicted_label = "defect" if anomaly_score > 0.3 else "good"
        true_label = class_names[labels.item()]
        is_correct = (predicted_label == "good" and true_label == "good") or \
                     (predicted_label == "defect" and true_label != "good")

        correct += int(is_correct)
        total += 1

        results.append({
            "orig": imgs_np,
            "recon": outputs_np,
            "anomaly_map": anomaly_map,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "is_correct": is_correct,
            "score": anomaly_score
        })

accuracy = correct / total * 100
print(f"\nPoprawnie sklasyfikowane: {correct}/{total} ({accuracy:.2f}%)")

# --- WIZUALIZACJA PRZYKŁADÓW ---
n_show = min(5, len(results))
plt.figure(figsize=(20, 10))
for i in range(n_show):
    r = results[i]
    plt.subplot(3, n_show, i+1)
    plt.imshow(r["orig"])
    plt.title(f"Oryginał: {r['true_label']}")
    plt.axis('off')

    plt.subplot(3, n_show, i+1+n_show)
    plt.imshow(r["recon"])
    plt.title(f"Rekonstrukcja\n(score={r['score']:.2f})")
    plt.axis('off')

    plt.subplot(3, n_show, i+1+2*n_show)
    plt.imshow(r["anomaly_map"], cmap='hot')
    color = "green" if r["is_correct"] else "red"
    plt.title(f"Pred: {r['predicted_label']}\n{'ok' if r['is_correct'] else 'nok'}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()