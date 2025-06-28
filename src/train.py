import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

final_df = pd.read_csv("../data/processed/final_with_target.csv")
X = final_df.drop(columns=["CustomerId", "is_high_risk"])
y = final_df["is_high_risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

mlflow.set_experiment("credit-risk-model")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

        mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
        mlflow.log_metric("precision", precision_score(y_test, preds))
        mlflow.log_metric("recall", recall_score(y_test, preds))
        mlflow.log_metric("f1_score", f1_score(y_test, preds))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, proba))

       
        mlflow.sklearn.log_model(model, name)

        print(f"{name} model logged to MLflow.")
