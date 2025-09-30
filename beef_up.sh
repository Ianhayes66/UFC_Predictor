#!/bin/bash
set -e

echo "📦 Installing extra dependencies..."
pip install requests beautifulsoup4

echo "📝 Creating UFC stats scraper..."
cat > model/scrape_ufcstats.py <<'EOF'
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "http://ufcstats.com/statistics/fighters?char={}&page=all"

def scrape_fighters():
    all_data = []
    for letter in list("abcdefghijklmnopqrstuvwxyz"):
        url = BASE_URL.format(letter)
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select("table.b-statistics__table tbody tr")
        for row in rows:
            cols = [c.get_text(strip=True) for c in row.select("td")]
            if not cols or len(cols) < 7:
                continue
            name, height, weight, reach, stance, wins, losses, draws = (
                cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7]
            )
            all_data.append({
                "name": name,
                "height": height,
                "weight": weight,
                "reach": reach,
                "stance": stance,
                "wins": wins,
                "losses": losses,
                "draws": draws
            })
    df = pd.DataFrame(all_data)
    df.to_csv("data/ufc_fighters.csv", index=False)
    print(f"Saved {len(df)} fighters to data/ufc_fighters.csv")

if __name__ == "__main__":
    scrape_fighters()
EOF

echo "⚙️ Updating train.py to handle feature importance..."
cat > model/train.py <<'EOF'
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
EOF

echo "🌐 Adding /feature-importance endpoint..."
cat >> app/main.py <<'EOF'

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
EOF

echo "✅ Beef-up complete! Steps to run:"
echo "1) python model/scrape_ufcstats.py   # get real fighter data"
echo "2) Replace training_data.csv with engineered features from your new data"
echo "3) python model/train.py"
echo "4) uvicorn app.main:app --reload --port 8000"
echo "5) Visit /feature-importance to see top features"
