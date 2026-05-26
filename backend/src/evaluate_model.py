import pandas as pd
import json

def evaluate_recommender():
    print("Memulai Evaluasi Model ML")

    df = pd.read_csv("data/bread_basket.csv")
    df.columns = df.columns.str.strip()

    df = df[df['Item'] != 'NONE']

    with open('models/knn_recommendations.json', 'r') as f:
        rekomendasi_model = json.load(f)

    transaksi_dict = df.groupby('Transaction')['Item'].apply(lambda x: list(set(x))).to_dict()
    transaksi_valid = {k: v for k, v in transaksi_dict.items() if len(v) >= 2}

    hits = 0
    total_precision = 0
    total_recall = 0
    total_eval = 0
    K = 4

    for tx_id, items in transaksi_valid.items():
        target_item = items[-1]
        input_items = items[:-1]

        hasil_prediksi = []
        for item in input_items:
            if item in rekomendasi_model:
                hasil_prediksi.extend(rekomendasi_model[item])

        prediksi_unik = []
        for p in hasil_prediksi:
            if p not in prediksi_unik and p not in input_items:
                prediksi_unik.append(p)

        top_k_prediksi = prediksi_unik[:K]

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
    
    hit_ratio_persen = (hits / total_eval) * 100
    precision_persen = (total_precision / total_eval) * 100
    recall_persen = (total_recall / total_eval) * 100

    print(f"Hit Ratio@{K} : {hit_ratio_persen:.2f}%")
    print(f"Precision@{K} : {precision_persen:.2f}%")
    print(f"Recall@{K}    : {recall_persen:.2f}%")

if __name__ == "__main__":
    evaluate_recommender()