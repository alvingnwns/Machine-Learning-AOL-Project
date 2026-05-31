from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from pathlib import Path

app = FastAPI(title="Cafe Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'knn_recommendations.json'
try:
    with open(MODEL_PATH, 'r', encoding='utf-8') as f:
        rekomendasi_model = json.load(f)
except FileNotFoundError as e:
    rekomendasi_model = {}
    print(f"File tidak ditemukan! Error: {e}")

class CartRequest(BaseModel):
    items: list[str]
    top_k: int = 4

@app.post("/recommend")
def get_recommendations(request: CartRequest):
    cart_items = request.items

    if not cart_items:
        return {"recommendations": []}

    # Score-weighted aggregation: accumulate similarity scores per candidate
    score_map: dict = {}
    for item in cart_items:
        if item in rekomendasi_model:
            for entry in rekomendasi_model[item]:
                neighbor, score = entry[0], entry[1]
                if neighbor not in cart_items:
                    score_map[neighbor] = score_map.get(neighbor, 0.0) + score

    top_k_prediksi = sorted(score_map, key=lambda x: score_map[x], reverse=True)[:request.top_k]

    return {
        "cart": cart_items,
        "recommendations": top_k_prediksi
    }