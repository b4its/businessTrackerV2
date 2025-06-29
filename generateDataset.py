import os
import json
import random
from datetime import datetime, timedelta

# Folder output
base_folder = "data"
normal_folder = os.path.join(base_folder, "normal")
norm_folder = os.path.join(base_folder, "normData")

os.makedirs(normal_folder, exist_ok=True)
os.makedirs(norm_folder, exist_ok=True)

# Konfigurasi dataset
kondisi_list = [
    "untung_besar",
    "rugi_besar",
    "seimbang",
    "modal_kecil",
    "modal_besar",
    "ekstrim_pemasukan",
    "ekstrim_pengeluaran"
]
file_per_kondisi = 20
data_per_file = 50

# Simpan semua data untuk normalisasi akhir
all_data = []

# Fungsi buat data berdasarkan kondisi
def generate_dataset(jumlah_data, kondisi, start_time):
    datasets = []

    for i in range(jumlah_data):
        jam = (start_time + timedelta(minutes=i)).isoformat()

        if kondisi == "untung_besar":
            modal_awal = random.randint(5_000_000, 10_000_000)
            pemasukan = random.randint(modal_awal + 3_000_000, modal_awal + 10_000_000)
            pengeluaran = random.randint(500_000, modal_awal // 2)

        elif kondisi == "rugi_besar":
            modal_awal = random.randint(5_000_000, 10_000_000)
            pengeluaran = random.randint(modal_awal + 3_000_000, modal_awal + 10_000_000)
            pemasukan = random.randint(500_000, modal_awal // 2)

        elif kondisi == "seimbang":
            modal_awal = random.randint(1_000_000, 10_000_000)
            selisih = random.randint(-100_000, 100_000)
            pengeluaran = random.randint(1_000_000, modal_awal)
            pemasukan = pengeluaran + selisih

        elif kondisi == "modal_kecil":
            modal_awal = random.randint(100_000, 500_000)
            pemasukan = random.randint(50_000, 1_000_000)
            pengeluaran = random.randint(50_000, 1_000_000)

        elif kondisi == "modal_besar":
            modal_awal = random.randint(10_000_000, 50_000_000)
            pemasukan = random.randint(5_000_000, 100_000_000)
            pengeluaran = random.randint(5_000_000, 100_000_000)

        elif kondisi == "ekstrim_pemasukan":
            modal_awal = random.randint(500_000, 5_000_000)
            pemasukan = random.randint(100_000_000, 500_000_000)
            pengeluaran = random.randint(100_000, 1_000_000)

        elif kondisi == "ekstrim_pengeluaran":
            modal_awal = random.randint(500_000, 5_000_000)
            pemasukan = random.randint(100_000, 1_000_000)
            pengeluaran = random.randint(100_000_000, 500_000_000)

        else:
            continue

        rugi = max(pengeluaran - pemasukan, 0)
        profit = pemasukan - pengeluaran if rugi == 0 else 0

        data = {
            "waktu": jam,
            "modal_awal": modal_awal,
            "total_pemasukan": pemasukan,
            "total_pengeluaran": pengeluaran,
            "rugi": rugi,
            "profit": profit
        }

        datasets.append(data)

    return datasets

# 1. Buat dan simpan data per kondisi
for kondisi in kondisi_list:
    kondisi_folder = os.path.join(normal_folder, kondisi)
    os.makedirs(kondisi_folder, exist_ok=True)

    for i in range(file_per_kondisi):
        file_path = os.path.join(kondisi_folder, f"dataset_{i+1}.json")

        # Tentukan start_time
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                existing_data = json.load(f)
            if existing_data:
                last_time_str = existing_data[-2]["waktu"]
                last_time = datetime.fromisoformat(last_time_str)
                start_time = last_time + timedelta(days=1)
            else:
                start_time = datetime.now()
        else:
            existing_data = []
            start_time = datetime.now()

        # Generate dan gabung data baru
        new_data = generate_dataset(data_per_file, kondisi, start_time)
        combined_data = existing_data + new_data

        with open(file_path, "w") as f:
            json.dump(combined_data, f, indent=2)

        all_data.extend(combined_data)

# 2. Cari nilai maksimum untuk normalisasi
max_values = {
    "modal_awal": max(d["modal_awal"] for d in all_data),
    "total_pemasukan": max(d["total_pemasukan"] for d in all_data),
    "total_pengeluaran": max(d["total_pengeluaran"] for d in all_data),
    "rugi": max((d["rugi"] for d in all_data if d["rugi"] > 0), default=1),
    "profit": max((d["profit"] for d in all_data if d["profit"] > 0), default=1),
}

# 3. Simpan dataset normalisasi
counter = 1
for kondisi in kondisi_list:
    kondisi_folder = os.path.join(normal_folder, kondisi)

    for file_name in os.listdir(kondisi_folder):
        full_path = os.path.join(kondisi_folder, file_name)

        with open(full_path) as f:
            data = json.load(f)

        norm_data = []
        for d in data:
            waktu = datetime.fromisoformat(d["waktu"])
            jam = waktu.hour / 24.0
            norm_data.append({
                "waktu": d["waktu"],
                "modal_awal": d["modal_awal"] / max_values["modal_awal"],
                "total_pemasukan": d["total_pemasukan"] / max_values["total_pemasukan"],
                "total_pengeluaran": d["total_pengeluaran"] / max_values["total_pengeluaran"],
                "rugi": d["rugi"] / max_values["rugi"],
                "profit": d["profit"] / max_values["profit"],
                "jam": jam
            })

        norm_file_path = os.path.join(norm_folder, f"dataset_{kondisi}_{counter}.json")
        with open(norm_file_path, "w") as f:
            json.dump(norm_data, f, indent=2)
        counter += 1

# 4. Simpan file statistik normalisasi
with open(os.path.join(norm_folder, "normalization_stats.json"), "w") as f:
    json.dump(max_values, f, indent=2)

print("✅ Dataset berhasil diperbarui dan dinormalisasi.")
