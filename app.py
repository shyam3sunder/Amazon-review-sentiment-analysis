from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import re
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64
import logging

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.DEBUG)

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("experiment.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        predictor = pickle.load(open(r"C:/Users/shyam/OneDrive/Desktop/amazon/Amazon review sentiment analysis/models/model_xgb.pkl", "rb"))
        scaler = pickle.load(open(r"C:/Users/shyam/OneDrive/Desktop/amazon/Amazon review sentiment analysis/models/scaler.pkl", "rb"))
        cv = pickle.load(open(r"C:/Users/shyam/OneDrive/Desktop/amazon/Amazon review sentiment analysis/models/countVectorizer.pkl", "rb"))

        if "file" in request.files:
            # Bulk prediction from CSV file
            file = request.files["file"]
            data = pd.read_csv(file)
            predictions, graph = bulk_prediction(predictor, scaler, cv, data)

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )
            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(graph.getbuffer()).decode("ascii")

            return response

        elif request.is_json and "text" in request.json:
            # Single string prediction
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
            return jsonify({"prediction": predicted_sentiment})

        else:
            raise ValueError("Invalid input: no file or text provided.")

    except Exception as e:
        logging.error("Error in /predict route", exc_info=True)
        return jsonify({"error": str(e)}), 500

def single_prediction(predictor, scaler, cv, text_input):
    try:
        corpus = []
        stemmer = PorterStemmer()
        review = re.sub("[^a-zA-Z]", " ", text_input).lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        corpus.append(" ".join(review))
        X_prediction = cv.transform(corpus).toarray()
        X_prediction_scl = scaler.transform(X_prediction)
        y_predictions = predictor.predict_proba(X_prediction_scl).argmax(axis=1)[0]
        return "Positive" if y_predictions == 1 else "Negative"
    except Exception as e:
        logging.error("Error in single_prediction function", exc_info=True)
        raise

def bulk_prediction(predictor, scaler, cv, data):
    try:
        corpus = []
        stemmer = PorterStemmer()
        for i in range(0, data.shape[0]):
            review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"]).lower().split()
            review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
            corpus.append(" ".join(review))

        X_prediction = cv.transform(corpus).toarray()
        X_prediction_scl = scaler.transform(X_prediction)
        y_predictions = predictor.predict_proba(X_prediction_scl).argmax(axis=1)
        data["Predicted sentiment"] = list(map(sentiment_mapping, y_predictions))
        predictions_csv = BytesIO()
        data.to_csv(predictions_csv, index=False)
        predictions_csv.seek(0)

        graph = get_distribution_graph(data)
        return predictions_csv, graph
    except Exception as e:
        logging.error("Error in bulk_prediction function", exc_info=True)
        raise

def get_distribution_graph(data):
    try:
        fig = plt.figure(figsize=(5, 5))
        colors = ("green", "red")
        wp = {"linewidth": 1, "edgecolor": "black"}
        tags = data["Predicted sentiment"].value_counts()
        explode = (0.01, 0.01)
        tags.plot(
            kind="pie",
            autopct="%1.1f%%",
            shadow=True,
            colors=colors,
            startangle=90,
            wedgeprops=wp,
            explode=explode,
            title="Sentiment Distribution"
        )
        graph = BytesIO()
        plt.savefig(graph, format="png")
        plt.close()
        return graph
    except Exception as e:
        logging.error("Error in get_distribution_graph function", exc_info=True)
        raise

def sentiment_mapping(x):
    return "Positive" if x == 1 else "Negative"

if __name__ == "__main__":
    app.run(port=5000, debug=True)


