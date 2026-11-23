
import pandas as pd
import pytest

from sklearn.dummy import DummyClassifier

from model_pipeline import (
    prepare_data,
    evaluate_model,
    save_model,
    load_model
)


def create_test_data(file_path):
    """Crée un petit fichier CSV pour les tests."""
    data = pd.DataFrame({
        "feature_1": list(range(20)),
        "feature_2": [i * 2 for i in range(20)],
        "readmit_30": [0, 1] * 10,
        "segment": ["A", "B"] * 10
    })

    data.to_csv(file_path, index=False)


def test_prepare_data(tmp_path):
    """Vérifie la préparation et la séparation des données."""
    file_path = tmp_path / "test_data.csv"
    create_test_data(file_path)

    X_train, X_test, y_train, y_test = prepare_data(file_path)

    assert len(X_train) == 16
    assert len(X_test) == 4
    assert "readmit_30" not in X_train.columns
    assert "segment" not in X_train.columns
    assert len(y_train) == 16
    assert len(y_test) == 4


def test_prepare_data_missing_column(tmp_path):
    """Vérifie la détection d'une colonne obligatoire absente."""
    file_path = tmp_path / "invalid_data.csv"

    data = pd.DataFrame({
        "feature_1": [1, 2, 3],
        "segment": ["A", "B", "A"]
    })

    data.to_csv(file_path, index=False)

    with pytest.raises(ValueError):
        prepare_data(file_path)


def test_save_and_load_model(tmp_path):
    """Vérifie la sauvegarde et le chargement d'un modèle."""
    X = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y = [0, 0, 1, 1]

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)

    file_path = tmp_path / "test_model.joblib"

    save_model(model, file_path)

    loaded_model = load_model(file_path)

    assert loaded_model.predict(X).tolist() == model.predict(X).tolist()


def test_evaluate_model():
    """Vérifie le calcul des métriques d'évaluation."""
    X_train = pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6]})
    y_train = [0, 0, 0, 1, 1, 1]

    model = DummyClassifier(strategy="prior")
    model.fit(X_train, y_train)

    X_test = pd.DataFrame({"feature": [7, 8, 9, 10]})
    y_test = [0, 1, 0, 1]

    models = {"Baseline": model}

    results = evaluate_model(models, X_test, y_test)

    assert "ROC-AUC" in results.columns
    assert "PR-AUC" in results.columns
    assert "F1" in results.columns
    assert "Baseline" in results.index
