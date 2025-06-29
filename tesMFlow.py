import os
import json
import torch
import joblib
import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from ctrain import BisnisAssistantModel  # pastikan file ctrain.py berisi class model yang sama

# ------------------- LOAD MLP MODEL ------------------- #
model = BisnisAssistantModel()
model.load_state_dict(torch.load("model/assistV1.pth"))
model.eval()

scaler_x = joblib.load("data/normData/scaler_x.pkl")
scaler_y = joblib.load("data/normData/scaler_y.pkl")

# ------------------- LOAD SBERT (jika perlu pengembangan NLP lanjut) ------------------- #
sbert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# ------------------- ENTITY EXTRACTION DARI PERTANYAAN ------------------- #
def extract_entities_dari_pertanyaan(text):
    text = text.lower()

    # Deteksi waktu
    if "hari ini" in text or "sekarang" in text:
        waktu = "hari_ini"
    elif "kemarin" in text:
        waktu = "kemarin"
    elif "minggu ini" in text:
        waktu = "minggu_ini"
    elif "bulan ini" in text:
        waktu = "bulan_ini"
    elif "tahun ini" in text:
        waktu = "tahun_ini"
    else:
        waktu = "all"  # fallback ke semua data

    # Deteksi target (intent)
    if "modal" in text:
        target = "modal"
    elif "rugi" in text or "kerugian" in text:
        target = "rugi"
    elif "untung" in text or "profit" in text or "laba" in text or "keuntungan" in text:
        target = "profit"
    else:
        target = "profit"  # default target

    return waktu, target

# ------------------- AMBIL DATA TRANSAKSI BERDASARKAN WAKTU ------------------- #
def ambil_data_by_waktu(waktu_target="hari_ini"):
    matched = []
    norm_dir = "data/normData"
    now = datetime.now()

    for file in os.listdir(norm_dir):
        if not file.endswith(".json") or file == "normalization_stats.json":
            continue

        with open(os.path.join(norm_dir, file)) as f:
            data = json.load(f)

        for item in data:
            try:
                waktu = datetime.fromisoformat(item["waktu"])
                if waktu_target == "hari_ini" and waktu.date() == now.date():
                    matched.append(item)
                elif waktu_target == "kemarin" and waktu.date() == (now.date() - timedelta(days=1)):
                    matched.append(item)
                elif waktu_target == "minggu_ini" and waktu.isocalendar()[1] == now.isocalendar()[1]:
                    matched.append(item)
                elif waktu_target == "bulan_ini" and waktu.month == now.month and waktu.year == now.year:
                    matched.append(item)
                elif waktu_target == "tahun_ini" and waktu.year == now.year:
                    matched.append(item)
                elif waktu_target == "all":
                    matched.append(item)
            except:
                continue

    if not matched:
        raise ValueError(f"Tidak ada data transaksi yang cocok untuk waktu: {waktu_target}")

    pemasukan = np.mean([item["total_pemasukan"] for item in matched])
    pengeluaran = np.mean([item["total_pengeluaran"] for item in matched])
    jam = np.mean([datetime.fromisoformat(item["waktu"]).hour / 24.0 for item in matched])

    return pemasukan, pengeluaran, jam

# ------------------- PREDIKSI DARI MODEL MLP ------------------- #
def predict_from_data(pemasukan, pengeluaran, jam_float):
    input_data = np.array([[pemasukan, pengeluaran, jam_float]], dtype=np.float32)
    input_scaled = scaler_x.transform(input_data)

    with torch.no_grad():
        pred_scaled = model(torch.tensor(input_scaled)).numpy()

    pred = scaler_y.inverse_transform(pred_scaled)[0]
    return {
        "modal": pred[0],
        "profit": pred[1],
        "rugi": pred[2]
    }

# ------------------- MAIN INTERACTIVE LOOP ------------------- #
if __name__ == "__main__":
    print("� Business Tracker Siap, silahkan tanyakan sesuatu...")
    print("� Contoh: 'Berapa profit saya hari ini?', 'Apakah saya rugi minggu ini?', dst.")
    print("Ketik 'exit' untuk keluar.\n")

    while True:
        try:
            user_input = input("Anda: ").strip()
            if user_input.lower() in ["exit", "quit", "keluar"]:
                print("� Sampai jumpa!")
                break

            # 1. Ekstrak waktu & target
            waktu, target_field = extract_entities_dari_pertanyaan(user_input)

            # 2. Ambil data
            pemasukan, pengeluaran, jam = ambil_data_by_waktu(waktu)

            # 3. Prediksi
            hasil = predict_from_data(pemasukan, pengeluaran, jam)

            # 4. Tampilkan
            print(f"� Prediksi untuk {waktu.replace('_', ' ')}:")
            print(f"→ {target_field.capitalize()} Anda diperkirakan sebesar Rp {hasil[target_field]:,.0f}\n")

        except Exception as e:
            print(f"⚠️ Terjadi kesalahan: {e}\n")
