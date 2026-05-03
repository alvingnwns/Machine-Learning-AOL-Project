import pandas as pd
from sklearn.neighbors import NearestNeighbors
import json
import os

def train_and_save_knn():
    print("Memulai proses training model KNN...")

    data_path = '../data/bread basket.csv'
    output_path = '../models/knn_recommendations.json'

    try:
        df = pd.read_csv(data_path)
        print("Dataset Loaded!")
    except:
        print(f"Error: File tidak ditemukan pada {data_path}")
        return
    
    df.columns = df.columns.str.strip()

    if 'Item' in df.columns:
        df = df[df['Item'] != 'NONE']

    print("Membangun matriks interaksi antar item-transaction...")
    item_transaction_matrix = pd.crosstab(df['Item'], df['Transaction']).clip(upper=1)

    print("Melatih model KNN...")
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
    knn.fit(item_transaction_matrix.values)

    rekomendasi_menu = {}

    for i, item in enumerate(item_transaction_matrix.index):
        distances, indices = knn.kneighbors(item_transaction_matrix.iloc[i, :].values.reshape(1, -1))

        rekomendasi = [item_transaction_matrix.index[idx] for idx in indices.flatten()[1:]]
        rekomendasi_menu[item] = rekomendasi
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(rekomendasi_menu, f, indent=4)

    print(f"Training KNN Selesai! Hasil tersimpan pada {output_path}")

if __name__ == "__main__":
    train_and_save_knn()