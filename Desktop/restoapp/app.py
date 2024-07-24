from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the ML model
model = joblib.load('/Users/rsoedarnadi/Downloads/catboost_model.pkl')

@app.route('/')
def home():
    return "Welcome to the ML model API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)