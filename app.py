from flask import Flask, jsonify, request
from flask_cors import CORS  # Import Flask-CORS
import os
import torch
from transformers import AutoModelForSequenceClassification, BertJapaneseTokenizer
import torch.nn.functional as F
from transformers import AutoConfig

app = Flask(__name__)
CORS(app)  # Kích hoạt CORS trên toàn bộ ứng dụng

# Đảm bảo rằng bạn sử dụng đường dẫn tương đối đúng
MODEL_PATH = os.path.join(os.getcwd(), 'checkpoint-250002')
CONFIG_PATH = os.path.join(MODEL_PATH, 'config.json').replace("\\", "/")
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
    threshold = float(data.get('threshold', 0.7))  # Ngưỡng mặc định là 0.7

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if not (0 <= threshold <= 1):
        return jsonify({"error": "Threshold must be between 0 and 1"}), 400

    load_model()

    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        
        prob_class_0 = probabilities[0][0].item()
        prob_class_1 = probabilities[0][1].item()

    prediction = "Correct" if prob_class_1 > threshold else "Incorrect"

    return jsonify({
        "input_text": text,
        "prediction": prediction,
        "probability_correct": prob_class_1,
        "probability_incorrect": prob_class_0,
        "threshold": threshold
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render cung cấp biến PORT
    app.run(host='0.0.0.0', port=port)
