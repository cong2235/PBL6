from flask import Flask, request, jsonify
from flask_cors import CORS
from elasticsearch import Elasticsearch
from waitress import serve
import logging
import os
import torch
from transformers import AutoModelForSequenceClassification, BertJapaneseTokenizer
import torch.nn.functional as F
from transformers import AutoConfig

app = Flask(__name__)
CORS(app)

# Elasticsearch configuration
es = Elasticsearch(
    hosts=["https://d93143eb81aa40ae9b186eeee81a1adc.us-central1.gcp.cloud.es.io"],
    basic_auth=("elastic", "YgZyYW1VLobvLzpWvVup0ZwE"),
    request_timeout=60
)

index_name = "japanese_sentences"

# Model configuration for grammar classification
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

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    return "Flask Server for Grammar Classification and Sentence Search"

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

@app.route('/search', methods=['POST'])
def search_and_compare():
    try:
        query = request.json.get('query')
        if not query:
            return jsonify({"error": "Query not provided"}), 400

        search_query = {
            "query": {
                "more_like_this": {
                    "fields": ["sentence"],
                    "like": query,
                    "min_term_freq": 1,
                    "max_query_terms": 12
                }
            },
            "size": 10  # Limit the number of documents retrieved
        }

        # Send search request to Elasticsearch
        response = es.search(index=index_name, body=search_query)

        results = []
        for hit in response['hits']['hits']:
            sentence = hit['_source']['sentence']
            score = hit['_score']
            results.append({"sentence": sentence, "score": score})

        # Sort results by score and get the highest score
        results = sorted(results, key=lambda x: x['score'], reverse=True)

        if results and results[0]['score'] > 35:
            return jsonify({"text": results[0]['sentence']})
        else:
            return jsonify({"text": "no similar sentence found"})
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render cung cấp biến PORT
    serve(app, host='0.0.0.0', port=port)
