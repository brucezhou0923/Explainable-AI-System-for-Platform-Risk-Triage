from pathlib import Path
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "platform_incident_risk_simulated.csv")

target = "escalation_required"
X = df.drop(columns=[target])
y = df[target]

num_cols = [
    "content_velocity", "user_reports_24h", "prior_violations",
    "model_confidence", "account_age_days", "engagement_spike_ratio"
]
cat_cols = ["cross_border_content", "content_type", "region_risk_band", "appeal_history"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=4,
    random_state=42
)

pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

metrics = {
    "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    "accuracy": round(float(accuracy_score(y_test, pred)), 4),
    "precision": round(float(precision_recall_fscore_support(y_test, pred, average="binary")[0]), 4),
    "recall": round(float(precision_recall_fscore_support(y_test, pred, average="binary")[1]), 4),
    "f1": round(float(precision_recall_fscore_support(y_test, pred, average="binary")[2]), 4),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
}

(BASE / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
