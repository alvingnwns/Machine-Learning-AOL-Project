from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

app = FastAPI(title="Cafe Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_PATH = '../models/knn_recommendations.json'
try:
    with open(MODEL_PATH, 'r') as f:
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
    
    hasil_prediksi = []

    for item in cart_items:
        if item in rekomendasi_model:
            hasil_prediksi.extend(rekomendasi_model[item])

    prediksi_unik = []
    for p in hasil_prediksi:
        if p not in prediksi_unik and p not in cart_items:
            prediksi_unik.append(p)

    top_k_prediksi = prediksi_unik[:request.top_k]

    return {
        "cart": cart_items,
        "recommendations": top_k_prediksi
    }