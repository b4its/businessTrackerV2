import json
from sentence_transformers import SentenceTransformer, util
import os

# Load model Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')  # Kamu bisa ganti ke model SBERT lain

# Data awal
original_data = [
    {
        "text": "Berapa keuntungan saya hari ini?",
        "intent": "tanya_profit",
        "entities": {
            "waktu": "hari_ini"
        }
    },
    {
        "text": "Apakah saya mengalami kerugian minggu ini?",
        "intent": "tanya_rugi",
        "entities": {
            "waktu": "minggu_ini"
        }
    }
]

# Mapping kata kunci ke sinonim dan entity
AUGMENT_ENTITIES = {
    "keuntungan": {
        "synonyms": ["profit", "pendapatan", "hasil"],
        "entity_key": "target",
        "entity_value": "keuntungan"
    },
    "kerugian": {
        "synonyms": ["rugi", "defisit", "negatif"],
        "entity_key": "target",
        "entity_value": "kerugian"
    },
    "hari ini": {
        "synonyms": ["sekarang", "per hari ini"],
        "entity_key": "waktu",
        "entity_value": "hari_ini"
    },
    "minggu ini": {
        "synonyms": ["7 hari terakhir", "sepekan ini"],
        "entity_key": "waktu",
        "entity_value": "minggu_ini"
    }
}

# Fungsi augmentasi pertanyaan berdasarkan sinonim
def augment_text(text, original_entities):
    augmented = []

    for phrase, info in AUGMENT_ENTITIES.items():
        if phrase in text:
            for synonym in info["synonyms"]:
                new_text = text.replace(phrase, synonym)
                new_entities = original_entities.copy()
                new_entities[info["entity_key"]] = info["entity_value"]

                augmented.append({
                    "text": new_text,
                    "entities": new_entities
                })

    return augmented

# Generate data baru
augmented_data = []

for item in original_data:
    base_text = item["text"]
    intent = item["intent"]
    base_entities = item["entities"]

    # Simpan data asli
    augmented_data.append({
        "text": base_text,
        "intent": intent,
        "entities": base_entities
    })

    # Augmentasi dengan sinonim
    augmented_variants = augment_text(base_text, base_entities)

    for variant in augmented_variants:
        augmented_data.append({
            "text": variant["text"],
            "intent": intent,
            "entities": variant["entities"]
        })

# Simpan ke file
os.makedirs("data", exist_ok=True)
with open("data/questions_augmentedv1.json", "w", encoding="utf-8") as f:
    json.dump(augmented_data, f, indent=2, ensure_ascii=False)

print("Augmentasi selesai. Disimpan di: data/questions_augmentedv1.json")
