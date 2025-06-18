
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import joblib

# Cargar el dataset
df = pd.read_csv("linguame_training_dataset_1000.csv")

# Separar características y etiquetas
X = df["text"]
y = df["label"]

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Definir el pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(pipeline, "linguame_model.pkl")

# Evaluación
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
