from flask import Flask, request, jsonify
from flask_cors import CORS
from tempfile import NamedTemporaryFile
import os
from classifier import classify

# import the classify_file helper from the test_rawnet module
from test_rawnet import classify_file

class VoiceAuthAPI:
    """Flask API that classifies uploaded WAVs as human or AI."""

    def __init__(self, model_path: str = "rawnet_classifier.pth", max_dur: float = 3.0):
        self.app = Flask(__name__)
        CORS(self.app)  # allow requests from all domains

        self.model_path = model_path
        self.max_dur = max_dur

        @self.app.route("/predict", methods=["POST"])
        def predict():
            if "file" not in request.files:
                return jsonify({"error": "file field missing"}), 400
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "empty filename"}), 400
            # save to temp file because classify_file expects a path
            with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name
            try:
                label = classify_file(tmp_path, model_path=self.model_path, max_duration=self.max_dur)
            except Exception as e:
                os.unlink(tmp_path)
                return jsonify({"error": str(e)}), 500
            os.unlink(tmp_path)
            return jsonify({"label": label})

        @self.app.route("/classify", methods=["POST"])
        def classify_route():
            print(request.files)
            if "audio1" not in request.files:
                return jsonify({"error": "No audio1 field"}), 400
            if "audio2" not in request.files:
                return jsonify({"error": "No audio2 field"}), 400
            result = classify(request.files["audio1"], request.files["audio2"])
            return jsonify(result)

    def run(self, host: str = "0.0.0.0", port: int = 5000):
        self.app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    api = VoiceAuthAPI(model_path="rawnet_classifier.pth", max_dur=3.0)
    api.run()