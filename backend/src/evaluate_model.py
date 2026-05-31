import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'bread_basket.csv'
MODEL_PATH = BASE_DIR / 'models' / 'knn_recommendations.json'
SPLIT_METADATA_PATH = BASE_DIR / 'models' / 'knn_split_metadata.json'

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


def deduplicate_preserving_order(items):
    return list(dict.fromkeys(items))

def evaluate_recommender():
    print("Memulai Evaluasi Model ML")

    df = load_dataset()

    with open(MODEL_PATH, 'r', encoding='utf-8') as f:
        rekomendasi_model = json.load(f)

    with open(SPLIT_METADATA_PATH, 'r', encoding='utf-8') as f:
        split_metadata = json.load(f)

    validation_ids = set(split_metadata['validation_transaction_ids'])
    validation_df = df[df['Transaction'].isin(validation_ids)].copy()

    transaksi_dict = validation_df.groupby('Transaction')['Item'].apply(
        lambda x: deduplicate_preserving_order(x.tolist())
    ).to_dict()
    transaksi_valid = {k: v for k, v in transaksi_dict.items() if len(v) >= 2}

    hits = 0
    total_precision = 0
    total_recall = 0
    total_eval = 0
    K = 4

    for tx_id, items in transaksi_valid.items():
        target_item = items[-1]
        input_items = items[:-1]

        # Score-weighted aggregation: accumulate similarity scores per candidate
        score_map: dict = {}
        for item in input_items:
            if item in rekomendasi_model:
                for entry in rekomendasi_model[item]:
                    neighbor, score = entry[0], entry[1]
                    if neighbor not in input_items:
                        score_map[neighbor] = score_map.get(neighbor, 0.0) + score

        # Sort by accumulated score descending
        top_k_prediksi = sorted(score_map, key=lambda x: score_map[x], reverse=True)[:K]

        if not top_k_prediksi:
            continue

        total_eval += 1

        if target_item in top_k_prediksi:
            hits += 1

        benar = 1 if target_item in top_k_prediksi else 0

        precision = benar / len(top_k_prediksi)
        total_precision += precision

        recall = benar / 1
        total_recall += recall

    if total_eval == 0:
        print("Tidak ada transaksi validation yang bisa dievaluasi.")
        return

    hit_ratio_persen = (hits / total_eval) * 100
    precision_persen = (total_precision / total_eval) * 100
    recall_persen = (total_recall / total_eval) * 100

    print(f"Validation transactions dievaluasi: {total_eval}")
    print(f"Hit Ratio@{K} : {hit_ratio_persen:.2f}%")
    print(f"Precision@{K} : {precision_persen:.2f}%")
    print(f"Recall@{K}    : {recall_persen:.2f}%")

if __name__ == "__main__":
    evaluate_recommender()