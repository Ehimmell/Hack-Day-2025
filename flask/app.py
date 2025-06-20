# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict

app = Flask(__name__)
CORS(app)                        # ← allows localhost:3000 to call us

@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file field"}), 400
    result = predict(request.files["file"])
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
