from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend to communicate with backend

print("Loading model and vectorizer...")
vectorization = joblib.load("vectorizer.pkl")
LR = joblib.load("LR_model.pkl")
print("Model and vectorizer loaded successfully!")

@app.route('/')
def home():
    return "Fake News Detection API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if "text" not in data:
            return jsonify({"error": "Invalid input"}), 400

        news_text = data["text"]
        new_xv_test = vectorization.transform([news_text])
        pred_LR = LR.predict(new_xv_test)

        return jsonify({"prediction": "Fake News" if pred_LR[0] == 0 else "Not Fake News"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
