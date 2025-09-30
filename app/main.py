from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "clf.pkl"

class Features(BaseModel):
    reach_diff: float = Field(..., description="Fighter reach minus opponent reach (in)")
    age_diff: float = Field(..., description="Fighter age minus opponent age (years)")
    height_diff: float = Field(..., description="Fighter height minus opponent height (in)")
    strike_acc_diff: float = Field(..., description="Significant strike accuracy diff (pp)")
    takedown_acc_diff: float = Field(..., description="Takedown accuracy diff (pp)")

app = FastAPI(title="UFC Win Probability (Toy)", version="0.1.0")

@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Train the model first.")
    model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(feats: Features):
    x = np.array([[
        feats.reach_diff,
        feats.age_diff,
        feats.height_diff,
        feats.strike_acc_diff,
        feats.takedown_acc_diff
    ]])
    proba = float(model.predict_proba(x)[0, 1])
    return {"win_probability": proba}

@app.get("/")
def root():
    return {"msg": "UFC Win Prob API. See /docs for Swagger."}

@app.get("/")
def root():
    return {"msg": "UFC Win Prob API. See /docs for Swagger."}

@app.get("/feature-importance")
def feature_importance():
    import numpy as np
    import joblib
    from pathlib import Path
    model = joblib.load(MODEL_PATH)
    features = Path(__file__).resolve().parents[1] / "model" / "features.txt"
    feature_names = features.read_text().splitlines()
    clf = model.named_steps["clf"]
    coefs = clf.coef_[0]
    return {
        "features": [
            {"name": name, "importance": float(weight)}
            for name, weight in zip(feature_names, coefs)
        ]
    }

@app.get("/feature-importance")
def feature_importance():
    import numpy as np
    import joblib
    from pathlib import Path
    model = joblib.load(MODEL_PATH)
    features = Path(__file__).resolve().parents[1] / "model" / "features.txt"
    feature_names = features.read_text().splitlines()
    clf = model.named_steps["clf"]
    coefs = clf.coef_[0]
    return {
        "features": [
            {"name": name, "importance": float(weight)}
            for name, weight in zip(feature_names, coefs)
        ]
    }

@app.get("/feature-importance")
def feature_importance():
    import numpy as np
    import joblib
    from pathlib import Path
    model = joblib.load(MODEL_PATH)
    features = Path(__file__).resolve().parents[1] / "model" / "features.txt"
    feature_names = features.read_text().splitlines()
    clf = model.named_steps["clf"]
    coefs = clf.coef_[0]
    return {
        "features": [
            {"name": name, "importance": float(weight)}
            for name, weight in zip(feature_names, coefs)
        ]
    }
