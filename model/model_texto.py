import re
import string
import joblib


# Cargar modelo y vectorizer
model = joblib.load("modelo/logistic_regression_model.pkl")
vectorizer = joblib.load("modelo/tfidf_vectorizer.pkl")


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def predice(text):
    print("=======***==== PREDECIR ====******")

    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)
    print(f"==========prediction: {prediction}")

    prediction_result = int(prediction[0])
    if prediction_result == 0:
        return "Fake News"
    elif prediction_result == 1:
        return "Not A Fake News"