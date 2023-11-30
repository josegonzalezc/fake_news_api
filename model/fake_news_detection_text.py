import re
import string
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score


def read_data(filename, **kwargs):
    raw_data = pd.read_csv(filename, **kwargs)
    return raw_data


news_df = read_data("data/dataset_curso.csv")
df = news_df.drop(['created_utc', 'id', 'image_url', 'linked_submission_id', 'num_comments', 'score', 'upvote_ratio'],
                  axis=1)
print(df.isnull().sum())

# Renombrar las columnas
df.rename(columns={'clean_title': 'text', '2_way_label': 'class'}, inplace=True)
print(df.head())


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


df["text"] = df["text"].apply(wordopt)

x = df["text"]
y = df["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

LR = LogisticRegression()
LR.fit(xv_train, y_train)

pred_lr = LR.predict(xv_test)

score = LR.score(xv_test, y_test)

print(score)

print(classification_report(y_test, pred_lr))
print(classification_report(y_test, pred_lr))

# Predicciones en forma de probabilidades para calcular log_loss y AUROC
prob_lr = LR.predict_proba(xv_test)

# Calculando la pérdida de log para la validación
val_loss = log_loss(y_test, prob_lr)
print(f'val loss: {val_loss:.4f}')

# Calculando la precisión de la validación
val_acc = accuracy_score(y_test, pred_lr)
print(f'val acc: {val_acc:.4f}')

# Calculando AUROC para la validación
# El método predict_proba() devuelve las probabilidades para cada clase.
# La segunda columna corresponde a la probabilidad de la clase '1'.
val_auroc = roc_auc_score(y_test, prob_lr[:, 1])
print(f'val auroc: {val_auroc:.4f}')

# # Serializar con joblib y Guardar el modelo y el vectorizador
joblib.dump(LR, "../modelo/logistic_regression_model.pkl")
joblib.dump(vectorization, "../modelo/tfidf_vectorizer.pkl")
