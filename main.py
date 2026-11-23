import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de Machine Learning pour la prédiction de réadmission"
    )

    parser.add_argument(
        "--step",
        choices=["prepare", "train", "evaluate", "load", "all"],
        default="all",
        help="Étape du pipeline à exécuter",
    )

    args = parser.parse_args()

    if args.step == "prepare":
        prepare_data()

    elif args.step == "train":
        X_train, X_test, y_train, y_test = prepare_data()
        models = train_model(X_train, y_train)
        results = evaluate_model(models, X_test, y_test)

        best_model_name = results["ROC-AUC"].idxmax()
        save_model(models[best_model_name])

    elif args.step == "evaluate":
        X_train, X_test, y_train, y_test = prepare_data()
        models = train_model(X_train, y_train)
        evaluate_model(models, X_test, y_test)

    elif args.step == "load":
        load_model()

    elif args.step == "all":
        X_train, X_test, y_train, y_test = prepare_data()
        models = train_model(X_train, y_train)
        results = evaluate_model(models, X_test, y_test)

        best_model_name = results["ROC-AUC"].idxmax()
        print("\nMeilleur modèle :", best_model_name)

        save_model(models[best_model_name])
        load_model()

    print("\nExécution terminée.")


if __name__ == "__main__":
    main()
