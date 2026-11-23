import argparse
import subprocess
import sys

from prefect import flow, task

from main import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

# ==========================================
# 1. TACHES DE CONTROLE DU CODE
# ==========================================


@task(name="Installation des dépendances")
def install_dependencies_task():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True,
    )


@task(name="Formatage du code avec Black")
def format_code_task():
    files = [
        "model_pipeline.py",
        "main.py",
        "pipeline_prefect.py",
    ]
    subprocess.run(
        [sys.executable, "-m", "black"] + files,
        check=True,
    )


@task(name="Vérification de la qualité avec Flake8")
def check_code_quality_task():
    files = [
        "model_pipeline.py",
        "main.py",
        "pipeline_prefect.py",
    ]
    subprocess.run(
        [sys.executable, "-m", "flake8", "--max-line-length=100"] + files,
        check=True,
    )


@task(name="Vérification de la sécurité avec Bandit")
def check_code_security_task():
    files = [
        "model_pipeline.py",
        "main.py",
        "pipeline_prefect.py",
    ]
    subprocess.run(
        [sys.executable, "-m", "bandit", "-lll", "-iii"] + files,
        check=True,
    )


@task(name="Exécution des tests unitaires avec Pytest")
def run_tests_task():
    subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "test_model_pipeline.py"],
        check=True,
    )


# ==========================================
# 2. TACHES MACHINE LEARNING
# ==========================================


@task(name="Préparation des données")
def prepare_data_task():
    return prepare_data()


@task(name="Entraînement du modèle")
def train_model_task(X_train, y_train):
    return train_model(X_train, y_train)


@task(name="Sauvegarde du modèle")
def save_model_task(models):
    best_model = models["Forêt aléatoire"]
    save_model(best_model)


@task(name="Chargement du modèle")
def load_model_task():
    return load_model()


@task(name="Évaluation du modèle")
def evaluate_model_task(models, X_test, y_test):
    return evaluate_model(models, X_test, y_test)


@task(name="Prédiction")
def prediction_task(model, X_test):
    predictions = model.predict(X_test)
    print("\nExemples de prédictions :")
    print(predictions[:10])
    return predictions


# ==========================================
# 3. DEFINITION DES FLOWS
# ==========================================


@flow(name="code")
def code_flow():
    install_dependencies_task()
    format_code_task()
    check_code_quality_task()
    check_code_security_task()
    run_tests_task()


@flow(name="all")
def all_flow():
    # Vérification du code avant le pipeline ML
    code_flow()

    X_train, X_test, y_train, y_test = prepare_data_task()

    models = train_model_task(X_train, y_train)

    save_model_task(models)

    results = evaluate_model_task(models, X_test, y_test)

    best_model = load_model_task()

    prediction_task(best_model, X_test)

    return results


@flow(name="train")
def train_flow():
    X_train, X_test, y_train, y_test = prepare_data_task()

    models = train_model_task(X_train, y_train)

    save_model_task(models)


@flow(name="evaluate")
def evaluate_flow():
    X_train, X_test, y_train, y_test = prepare_data_task()

    model = load_model_task()

    models = {"Forêt aléatoire": model}

    results = evaluate_model_task(models, X_test, y_test)

    prediction_task(model, X_test)

    return results


# ==========================================
# 4. LANCEMENT DEPUIS LE TERMINAL
# ==========================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pipeline Machine Learning avec Prefect"
    )

    parser.add_argument(
        "--flow",
        choices=["all", "code", "entrainement", "train", "evaluate"],
        default="all",
        help="Flow à exécuter",
    )

    args = parser.parse_args()

    if args.flow == "all":
        all_flow()

    elif args.flow == "code":
        code_flow()

    elif args.flow in ["entrainement", "train"]:
        train_flow()

    elif args.flow == "evaluate":
        evaluate_flow()
