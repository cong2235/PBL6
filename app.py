import os
import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertJapaneseTokenizer
import json
from transformers import AutoConfig
import threading

app = Flask(__name__)

# Đảm bảo rằng bạn sử dụng đường dẫn tương đối đúng
MODEL_PATH = os.path.join(os.getcwd(), 'checkpoint-250002')
CONFIG_PATH = os.path.join(MODEL_PATH, 'config.json').replace("\\", "/")
MODEL_FILE = os.path.join(MODEL_PATH, 'model.safetensors')

LABELS = ['True', 'False']

model = None
tokenizer = None

def load_model():
    """Tải mô hình và tokenizer"""
    global model, tokenizer
    if model is None:
        config = AutoConfig.from_pretrained(CONFIG_PATH)
        tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, config=config)

@app.route('/')
def home():
    return "Flask Server for Grammar Classification"

@app.route('/predict', methods=['POST'])
def predict():
    """API nhận câu và trả về dự đoán đúng/sai ngữ pháp"""
    data = request.get_json()
    text = data.get('text', None)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    load_model()

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction_idx = torch.argmax(logits, dim=-1).item()
        prediction_label = LABELS[prediction_idx]

    return jsonify({
        "input_text": text,
        "prediction": prediction_label
    })

@app.route('/reload_model', methods=['POST'])
def reload_model():
    """API để tải lại mô hình"""
    load_model()
    return jsonify({"status": "Model reloaded successfully!"})

def run():
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)  # `use_reloader=False` tránh việc Flask tự khởi động lại khi chạy trong Jupyter

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render cung cấp biến PORT
    app.run(host='0.0.0.0', port=port)
