import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'bread_basket.csv'
MODEL_PATH = BASE_DIR / 'models' / 'knn_recommendations.json'
SPLIT_METADATA_PATH = BASE_DIR / 'models' / 'knn_split_metadata.json'
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

CHOSEN_ITEMS = [
    'Bread', 'Salad', 'Hot chocolate', 'Jam', 'Cookies', 'Muffin', 'Coffee', 'Pastry',
    'Medialuna', 'Tea', 'Tartine', 'Basket', 'Mineral water', 'Fudge', 'Juice',
    'Victorian Sponge', 'Frittata', 'Soup', 'Smoothies', 'Cake', 'Coke', 'Sandwich',
    'Baguette', 'Eggs', 'Brownie', 'Bread Pudding', 'Bacon', 'Toast', 'Scone', 'Crepes'
]


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df = df[df['Item'] != 'NONE']
    return df[df['Item'].isin(CHOSEN_ITEMS)].copy()


def split_by_transaction(df: pd.DataFrame):
    transaction_ids = df['Transaction'].drop_duplicates().tolist()
    train_ids, validation_ids = train_test_split(
        transaction_ids,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )
    train_df = df[df['Transaction'].isin(train_ids)].copy()
    return train_df, sorted(validation_ids)

def train_and_save_knn():
    print("Memulai proses training model KNN...")

    try:
        df = load_dataset()
        print("Dataset Loaded!")
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan pada {DATA_PATH}")
        return

    print(f"Filter diterapkan: {df['Item'].nunique()} item unik tersisa dari 30 item yang dipilih.")

    train_df, validation_ids = split_by_transaction(df)
    print(
        "Split transaksi selesai: "
        f"train={train_df['Transaction'].nunique()} transaksi, "
        f"validation={len(validation_ids)} transaksi."
    )

    print("Membangun matriks interaksi antar item-transaction...")
    # .clip(upper=1) sebagai pengubah aturan menjadi biner (0 dan 1)
    item_transaction_matrix = pd.crosstab(train_df['Item'], train_df['Transaction']).clip(upper=1)

    if item_transaction_matrix.empty:
        print("Error: Matriks interaksi kosong setelah split training.")
        return

    print("Melatih model KNN...")
    knn = NearestNeighbors(
        metric='cosine',
        algorithm='brute',
        n_neighbors=min(5, len(item_transaction_matrix.index)),
    )
    knn.fit(item_transaction_matrix.values)

    rekomendasi_menu = {}

    for i, item in enumerate(item_transaction_matrix.index):
        distances, indices = knn.kneighbors(item_transaction_matrix.iloc[i, :].values.reshape(1, -1))

        # Simpan sebagai list of [neighbor, similarity_score] agar ranking berbasis evidence
        rekomendasi = [
            [item_transaction_matrix.index[idx], round(1 - float(d), 6)]
            for idx, d in zip(indices.flatten()[1:], distances.flatten()[1:])
        ]
        rekomendasi_menu[item] = rekomendasi

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, 'w', encoding='utf-8') as f:
        json.dump(rekomendasi_menu, f, indent=4)

    with open(SPLIT_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'random_state': RANDOM_STATE,
                'validation_size': VALIDATION_SIZE,
                'validation_transaction_ids': validation_ids,
            },
            f,
            indent=4,
        )

    print(f"Training KNN selesai! Hasil tersimpan pada {MODEL_PATH}")
    print(f"Metadata split tersimpan pada {SPLIT_METADATA_PATH}")

if __name__ == "__main__":
    train_and_save_knn()