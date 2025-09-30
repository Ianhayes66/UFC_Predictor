from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import numpy as np

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "training_data.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "clf.pkl"
FEATURES_PATH = Path(__file__).resolve().parents[1] / "model" / "features.txt"

def main():
    df = pd.read_csv(DATA_PATH)
    y = df["win"]
    X = df.drop(columns=["win"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=200)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    print("AUC:", roc_auc_score(y_test, y_proba))
    print("ACC:", accuracy_score(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    FEATURES_PATH.write_text("\n".join(X.columns))
    print("Saved model to", MODEL_PATH)

if __name__ == "__main__":
    main()
