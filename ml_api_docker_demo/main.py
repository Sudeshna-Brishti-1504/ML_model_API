from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

app = FastAPI(title="ML Training and Prediction API")

MODEL_PATH = "iris_model.pkl"


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {
        "message": "ML API is running",
        "endpoints": ["/train", "/test", "/predict"]
    }
@app.post("/train")
def train_model():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump({
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "target_names": iris.target_names
    }, MODEL_PATH)

    return {
        "message": "Model trained successfully",
        "training_samples": len(X_train),
        "testing_samples": len(X_test)
    }


@app.get("/test")
def test_model():
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not trained. Please call /train first."}

    saved_data = joblib.load(MODEL_PATH)

    model = saved_data["model"]
    X_test = saved_data["X_test"]
    y_test = saved_data["y_test"]

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return {
        "message": "Model tested successfully",
        "accuracy": accuracy
    }


@app.post("/predict")
def predict(input_data: IrisInput):
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not trained. Please call /train first."}

    saved_data = joblib.load(MODEL_PATH)

    model = saved_data["model"]
    target_names = saved_data["target_names"]
    
    data = [input_data.sepal_length, input_data.sepal_width, input_data.petal_length, input_data.petal_width]

    prediction = model.predict(data)[0]

    return {
        "prediction_class": int(prediction),
        "flower_name": target_names[prediction]
    }