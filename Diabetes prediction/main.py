from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("diabetes.csv")
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "diabetes_model.joblib"
FEATURE_PLOT_PATH = ARTIFACTS_DIR / "feature_distributions.png"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.png"

TARGET_COLUMN = "Outcome"
RANDOM_STATE = 42


# In the Pima diabetes dataset, zero is used as a placeholder for missing
# clinical measurements in these columns.
ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the diabetes dataset and split it into features and target."""
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset: {path}")

    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column.")

    df[ZERO_AS_MISSING_COLUMNS] = df[ZERO_AS_MISSING_COLUMNS].replace(0, np.nan)
    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def build_candidate_models() -> dict[str, Pipeline]:
    """Create ML pipelines to compare."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        class_weight="balanced",
                        n_estimators=300,
                        max_depth=5,
                        min_samples_leaf=4,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "neural_network": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(12,12),
                        activation="relu",
                        solver="lbfgs",
                        alpha=0.1,
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def select_best_model(
    models: dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[str, Pipeline]:
    """Choose the model with the best cross-validated ROC AUC."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = {}

    print("\nCross-validation ROC AUC:")
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        scores[name] = cv_scores.mean()
        print(f"  {name:20s} {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    best_name = max(scores, key=scores.get)
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    return best_name, best_model


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> None:
    """Print final test metrics and save a confusion matrix image."""
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    print(f"\nBest model: {model_name}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Test ROC AUC:  {roc_auc_score(y_test, y_probability):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["No diabetes", "Diabetes"]))

    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No diabetes", "Diabetes"],
    )
    display.plot(values_format="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close()


def save_feature_distribution_plot(X: pd.DataFrame) -> None:
    """Save histograms for a quick visual check of the feature data."""
    axes = X.hist(figsize=(12, 9), bins=25)
    for axis in axes.ravel():
        axis.set_xlabel("")
        axis.set_ylabel("Count")

    plt.suptitle("Diabetes Dataset Feature Distributions", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(FEATURE_PLOT_PATH, dpi=160, bbox_inches="tight")
    plt.close()


def predict_patient_risk(model: Pipeline, patient_data: dict[str, float]) -> float:
    """Return diabetes probability for one patient record."""
    patient_df = pd.DataFrame([patient_data])
    return float(model.predict_proba(patient_df)[0, 1])


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    X, y = load_dataset()
    print("Dataset shape:", X.shape)
    print("Target counts:")
    print(y.value_counts().rename(index={0: "No diabetes", 1: "Diabetes"}))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = build_candidate_models()
    best_name, best_model = select_best_model(models, X_train, y_train)
    evaluate_model(best_model, X_test, y_test, best_name)

    joblib.dump(best_model, MODEL_PATH)
    save_feature_distribution_plot(X)

    sample_patient = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 80,
        "BMI": 32.0,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 35,
    }
    sample_probability = predict_patient_risk(best_model, sample_patient)

    print("\nSaved artifacts:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Feature distributions: {FEATURE_PLOT_PATH}")
    print(f"  Confusion matrix: {CONFUSION_MATRIX_PATH}")
    print(f"\nSample patient diabetes probability: {sample_probability:.1%}")


if __name__ == "__main__":
    main()
