from sentence_transformers import SentenceTransformer
import json
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

# Sinonim dan entitas yang terkait
AUGMENT_ENTITIES = {
    "keuntungan": {
        "synonyms": ["profit", "pendapatan", "hasil"],
        "entity_key": "target",
        "entity_value": "keuntungan"
    },
    "kerugian": {
        "synonyms": ["rugi", "defisit"],
        "entity_key": "target",
        "entity_value": "kerugian"
    },
    "modal": {
        "synonyms": ["dana awal", "investasi awal"],
        "entity_key": "target",
        "entity_value": "modal"
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

def generate_augmented_questions(text, original_entities):
    augmented_pairs = []

    for key, info in AUGMENT_ENTITIES.items():
        if key in text:
            for synonym in info["synonyms"]:
                new_text = text.replace(key, synonym)
                new_entities = original_entities.copy()

                # Perbarui entitas berdasarkan synonym
                new_entities[info["entity_key"]] = info["entity_value"]
                augmented_pairs.append((new_text, new_entities))

    return augmented_pairs

# === Load original data ===
with open("data/questions_augmentedv1.json", "r") as f:
    dataset = json.load(f)

augmented_dataset = []

for entry in dataset:
    original_text = entry["text"]
    intent = entry["intent"]
    entities = entry.get("entities", {})

    # Simpan data asli
    augmented_dataset.append({
        "text": original_text,
        "intent": intent,
        "entities": entities
    })

    # Augmentasi berdasarkan sinonim
    augmentations = generate_augmented_questions(original_text, entities)

    for aug_text, new_entities in augmentations:
        augmented_dataset.append({
            "text": aug_text,
            "intent": intent,
            "entities": new_entities
        })

# === Simpan hasil ===
os.makedirs("data", exist_ok=True)
with open("data/questions_augmentedv2.json", "w") as f:
    json.dump(augmented_dataset, f, indent=2, ensure_ascii=False)
