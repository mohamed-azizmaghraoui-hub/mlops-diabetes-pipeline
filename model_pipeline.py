import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
)

RANDOM_STATE = 42


def prepare_data(file_path="diabetic_data_prepared.csv"):
    """
    Charge les données préparées et crée les jeux d'apprentissage et de test.
    """
    data = pd.read_csv(file_path)

    if "readmit_30" not in data.columns:
        raise ValueError("La colonne readmit_30 est absente du fichier.")

    if "segment" not in data.columns:
        raise ValueError("La colonne segment est absente du fichier.")

    X = data.drop(columns=["readmit_30", "segment"])
    y = data["readmit_30"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("Données chargées :", data.shape)
    print("Entraînement :", X_train.shape)
    print("Test :", X_test.shape)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Entraîne quatre modèles de classification et une baseline.
    La recherche des hyperparamètres utilise une validation croisée.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    searches = {
        "Régression logistique": GridSearchCV(
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            class_weight="balanced", max_iter=3000, solver="lbfgs"
                        ),
                    ),
                ]
            ),
            {"clf__C": [0.001, 0.01, 0.1, 1]},
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        ),
        "Arbre de décision": GridSearchCV(
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"max_depth": [4, 6, 8, 12], "min_samples_leaf": [50, 200]},
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        ),
        "Forêt aléatoire": GridSearchCV(
            RandomForestClassifier(
                n_estimators=250,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            {"max_depth": [10, 16], "min_samples_leaf": [10, 40]},
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
        ),
        "Gradient Boosting": GridSearchCV(
            HistGradientBoostingClassifier(
                class_weight="balanced",
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=RANDOM_STATE,
            ),
            {"learning_rate": [0.03, 0.1], "max_depth": [3, 6]},
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        ),
    }

    models = {
        "Baseline (majoritaire)": DummyClassifier(strategy="most_frequent").fit(
            X_train, y_train
        )
    }

    for name, search in searches.items():
        print(f"\nEntraînement : {name}")
        search.fit(X_train, y_train)
        models[name] = search.best_estimator_

        print("Meilleurs paramètres :", search.best_params_)
        print("ROC-AUC CV :", round(search.best_score_, 4))

    return models


def evaluate_model(models, X_test, y_test):
    """
    Évalue les modèles sur le jeu de test.
    """
    rows = []

    for name, model in models.items():
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        rows.append(
            {
                "Modèle": name,
                "ROC-AUC": roc_auc_score(y_test, probabilities),
                "PR-AUC": average_precision_score(y_test, probabilities),
                "Rappel": recall_score(y_test, predictions),
                "Précision": precision_score(y_test, predictions, zero_division=0),
                "F1": f1_score(y_test, predictions),
                "Balanced accuracy": balanced_accuracy_score(y_test, predictions),
                "Accuracy": accuracy_score(y_test, predictions),
            }
        )

    results = pd.DataFrame(rows).set_index("Modèle")

    print("\nRésultats de l'évaluation :")
    print(results.round(3))

    return results


def save_model(model, file_path="best_model.joblib"):
    """
    Sauvegarde un modèle avec joblib.
    """
    joblib.dump(model, file_path)
    print("Modèle sauvegardé dans :", file_path)


def load_model(file_path="best_model.joblib"):
    """
    Charge un modèle sauvegardé.
    """
    model = joblib.load(file_path)
    print("Modèle chargé depuis :", file_path)
    return model
