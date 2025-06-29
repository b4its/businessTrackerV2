import os
import json
import torch
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from sentence_transformers import SentenceTransformer, util

# ------------------ DEFINISI MODEL ------------------ #
class BisnisAssistantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.model(x)

# ------------------ PERSIAPAN DATA ------------------ #
X, y = [], []

normal_dir = "data/normal"
norm_dir = "data/normData"
os.makedirs(norm_dir, exist_ok=True)

for label_folder in os.listdir(normal_dir):
    label_path = os.path.join(normal_dir, label_folder)
    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(label_path, file)) as f:
            data = json.load(f)

        out_file = os.path.join(norm_dir, f"dataset_{label_folder}_{file}")
        with open(out_file, "w") as out:
            json.dump(data, out, indent=2)

        for item in data:
            pemasukan = item["total_pemasukan"]
            pengeluaran = item["total_pengeluaran"]
            waktu = datetime.fromisoformat(item["waktu"])
            jam = waktu.hour / 24.0

            X.append([pemasukan, pengeluaran, jam])
            modal = item["modal_awal"]
            rugi = item["rugi"]
            profit = pemasukan - pengeluaran if rugi == 0 else 0
            y.append([modal, profit, rugi])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

joblib.dump(scaler_x, os.path.join(norm_dir, "scaler_x.pkl"))
joblib.dump(scaler_y, os.path.join(norm_dir, "scaler_y.pkl"))

with open(os.path.join(norm_dir, "normalization_stats.json"), "w") as f:
    json.dump({"x_max": scaler_x.data_max_.tolist(), "y_max": scaler_y.data_max_.tolist()}, f, indent=2)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# ------------------ TRAINING MODEL + MLFLOW ------------------ #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BisnisAssistantModel().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

os.makedirs("model", exist_ok=True)

patience = 20
min_delta = 1e-4
best_loss = float("inf")
wait = 0

# Set tracking URI dan experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment_name = "bisnis-trackerV1"
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    pbar = tqdm(range(1000), desc="Training")

    for epoch in pbar:
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")

        if epoch_loss + min_delta < best_loss:
            best_loss = epoch_loss
            wait = 0
            torch.save(model.state_dict(), "model/assistV1.pth")
        else:
            wait += 1
            if wait >= patience:
                pbar.close()
                print(f"\n⏹ Early stopping triggered at epoch {epoch}")
                break

    # Logging ke MLflow
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("patience", patience)
    mlflow.log_metric("final_train_loss", best_loss)

    example_input = np.random.randn(1, 3).astype(np.float32)
    mlflow.pytorch.log_model(
        model,
        artifact_path="bisnisTrackerV1",
        input_example=example_input
    )

    print("✅ Model berhasil dicatat ke MLflow")


# ------------------ TEXT PARAPHRASING SBERT ------------------ #
INTENT_TEMPLATES = {
    "tanya_profit": {
        "templates": [
            "Berapa keuntungan saya hari ini?",
            "Berapakah profit saya hari ini?",
            "Apa laba saya per hari ini?",
            "Keuntungan saya hari ini berapa?",
            "Apakah saya mendapatkan keuntungan hari ini?"
        ],
        "entities": {
            "waktu": "hari_ini",
            "target": "keuntungan"
        }
    },
    "tanya_rugi": {
        "templates": [
            "Apakah saya mengalami kerugian minggu ini?",
            "Saya rugi minggu ini?",
            "Rugi saya minggu ini berapa?",
            "Apakah saya mengalami rugi sepekan ini?",
            "Apakah saya mengalami defisit sepekan ini?"
        ],
        "entities": {
            "waktu": "minggu_ini",
            "target": "kerugian"
        }
    }
}

text_data_paths = [
    "data/questions_augmentedv1.json",
    "data/questions_augmentedv2.json"
]

text_data = []
for path in text_data_paths:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            text_data.extend(json.load(f))
    else:
        print(f"⚠️ File tidak ditemukan: {path}")

model_sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
augmented_data = []

for entry in text_data:
    question = entry["text"]
    intent = entry["intent"]

    if intent not in INTENT_TEMPLATES:
        continue

    templates = INTENT_TEMPLATES[intent]["templates"]
    base_entities = INTENT_TEMPLATES[intent]["entities"]

    try:
        question_embedding = model_sbert.encode(question, convert_to_tensor=True)
        template_embeddings = model_sbert.encode(templates, convert_to_tensor=True)
        cos_scores = util.cos_sim(question_embedding, template_embeddings)[0]
        top_indices = cos_scores.argsort(descending=True)[:3]

        for idx in top_indices:
            new_question = templates[idx]
            augmented_data.append({
                "text": new_question,
                "intent": intent,
                "entities": base_entities
            })
    except Exception as e:
        print(f"Paraphrase error: {e}")

for entry in text_data:
    if "entities" not in entry:
        entry["entities"] = INTENT_TEMPLATES.get(entry["intent"], {}).get("entities", {})

text_data.extend(augmented_data)

with open("data/questions_augmentedv2.json", "w", encoding="utf-8") as f:
    json.dump(text_data, f, indent=2, ensure_ascii=False)

# print("✅ Pertanyaan augmented disimpan ke 'data/questions_augmentedv2.json'")
