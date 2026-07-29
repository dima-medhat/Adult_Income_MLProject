from pathlib import Path

import joblib


MODEL_FILES = {
    "Naive Bayes": {
        "model": "naive_bayes_tuned.pkl",
        "results": "naive_bayes_results.pkl",
        "search": "naive_bayes_search.pkl",
    },
    "Decision Tree": {
        "model": "decision_tree_tuned.pkl",
        "results": "decision_tree_results.pkl",
        "search": "decision_tree_search.pkl",
    },
    "Random Forest": {
        "model": "random_forest_tuned.pkl",
        "results": "random_forest_results.pkl",
        "search": "random_forest_search.pkl",
    },
    "SVM": {
        "model": "svm_tuned.pkl",
        "results": "svm_results.pkl",
        "search": "svm_search.pkl",
    },
    "XGBoost": {
        "model": "xgboost_tuned.pkl",
        "results": "xgboost_results.pkl",
        "search": "xgboost_search.pkl",
    },
}


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def get_file_path(model_name, file_type):
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unsupported model: {model_name}")

    if file_type not in {"model", "results", "search"}:
        raise ValueError(f"Unsupported file type: {file_type}")

    return MODELS_DIR / MODEL_FILES[model_name][file_type]


def file_exists(model_name, file_type):
    return get_file_path(model_name, file_type).exists()


def load_model(model_name):
    model_path = get_file_path(model_name, "model")

    if not model_path.exists():
        return None

    return joblib.load(model_path)


def load_results(model_name):
    results_path = get_file_path(model_name, "results")

    if not results_path.exists():
        return None

    return joblib.load(results_path)


def load_search(model_name):
    search_path = get_file_path(model_name, "search")

    if not search_path.exists():
        return None

    return joblib.load(search_path)