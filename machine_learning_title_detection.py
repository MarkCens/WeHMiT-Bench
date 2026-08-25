# Import command-line, serialization, regular-expression, and path utilities.
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

# Use interpretable scikit-learn text models and shared evaluation metrics.
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# Share the flat artifact directory used by all other experiments.
DEFAULT_OUTPUT_DIR = "experiment_outputs"


def parse_args():
    # Configure fixed data splits, feature capacity, uncertainty, and model export.
    parser = argparse.ArgumentParser(description="Train and evaluate machine learning classifiers for title detection.")
    parser.add_argument("--train-file", default=f"{DEFAULT_OUTPUT_DIR}/01_train.csv", help="Training-set CSV.")
    parser.add_argument("--dev-file", default=f"{DEFAULT_OUTPUT_DIR}/01_dev.csv", help="Development-set CSV.")
    parser.add_argument("--test-file", default=f"{DEFAULT_OUTPUT_DIR}/01_test.csv", help="Test-set CSV.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for classifier outputs.")

    # Separate model capacity from bootstrap and persistence controls.
    parser.add_argument("--max-features", type=int, default=80000, help="Maximum TF-IDF character features.")
    parser.add_argument("--bootstrap-rounds", type=int, default=1000, help="Bootstrap rounds for confidence intervals.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-models", action="store_true", help="Save fitted scikit-learn pipelines.")
    parser.add_argument("--force", action="store_true", help="Refit classifiers even when inputs and settings are unchanged.")
    return parser.parse_args()


def load_split(file_path):
    # Read one preprocessing split and verify the minimal supervised schema.
    data_frame = pd.read_csv(file_path, encoding="utf-8-sig")
    required_columns = {"id", "title", "gold_label"}
    missing_columns = required_columns - set(data_frame.columns)
    if missing_columns:
        raise ValueError(f"{file_path} is missing columns: {sorted(missing_columns)}")
    return data_frame


def make_pipeline(model, max_features):
    # Character n-grams handle Chinese text without external word segmentation.
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, max_df=0.98, max_features=max_features, sublinear_tf=True)
    return Pipeline([("tfidf", vectorizer), ("classifier", model)])


def rule_predict(title):
    # Separate negation, high-risk rhetoric, and explicit medical claim patterns.
    text = str(title)
    negative_pattern = r"辟谣|别信|勿信|莫信|不靠谱|骗局|警惕|提醒|误区|新闻|实录|指南|科普"
    high_risk_pattern = r"根治|断根|永不复发|神效|奇效|特效|百试百灵|一招见效|立竿见影|秘方|祖传|包治|专治|不用吃药|不吃药|不用手术|不用化疗|嗖嗖"
    medical_claim_pattern = r"治疗|治愈|降血压|降血糖|降血脂|降尿酸|抗癌|排毒|清血管|提高免疫力|增强免疫力"

    # Explicit debunking overrides isolated trigger words in this transparent baseline.
    if re.search(negative_pattern, text):
        return "No", 0.10
    if re.search(r"[?？]", text) and not re.search(r"根治|专治|永不复发|神效", text):
        return "No", 0.25

    # Give deterministic positive scores to strong or combined claim patterns.
    if re.search(high_risk_pattern, text):
        return "Yes", 0.90
    if re.search(r"偏方|食疗|药膳|保健品|补充剂", text) and re.search(medical_claim_pattern, text):
        return "Yes", 0.75
    return "No", 0.30


def build_candidate_models(seed):
    candidate_models = []

    # Hyperparameters are intentionally small because the dataset contains hundreds of titles.
    for c_value in [0.25, 1.0, 4.0]:
        estimator = LogisticRegression(C=c_value, class_weight="balanced", max_iter=3000, random_state=seed)
        candidate_models.append(("char_tfidf_logistic", {"C": c_value}, estimator))

    # Apply the same small regularization grid to a linear margin classifier.
    for c_value in [0.25, 1.0, 4.0]:
        estimator = LinearSVC(C=c_value, class_weight="balanced", random_state=seed)
        candidate_models.append(("char_tfidf_linear_svm", {"C": c_value}, estimator))

    # Complement Naive Bayes provides a lightweight probabilistic text baseline.
    for alpha_value in [0.1, 0.5, 1.0]:
        estimator = ComplementNB(alpha=alpha_value)
        candidate_models.append(("char_tfidf_complement_nb", {"alpha": alpha_value}, estimator))
    return candidate_models


def select_hyperparameters(train_data, dev_data, max_features, seed):
    # Materialize title and label series once for every candidate configuration.
    selection_rows = []
    best_configs = {}
    train_titles = train_data["title"].astype(str)
    train_labels = train_data["gold_label"].astype(str)
    dev_titles = dev_data["title"].astype(str)
    dev_labels = dev_data["gold_label"].astype(str)

    # Select one configuration per model family using development macro F1.
    for model_name, parameters, estimator in build_candidate_models(seed):
        pipeline = make_pipeline(estimator, max_features)
        pipeline.fit(train_titles, train_labels)
        dev_predictions = pipeline.predict(dev_titles)
        macro_f1 = f1_score(dev_labels, dev_predictions, average="macro", zero_division=0)

        # Retain all development scores while tracking each family's best setting.
        selection_rows.append({"model_name": model_name, "parameters": json.dumps(parameters, ensure_ascii=False, sort_keys=True), "dev_macro_f1": float(macro_f1)})
        if model_name not in best_configs or macro_f1 > best_configs[model_name]["dev_macro_f1"]:
            best_configs[model_name] = {"parameters": parameters, "dev_macro_f1": float(macro_f1)}
    return best_configs, pd.DataFrame(selection_rows)


def make_estimator(model_name, parameters, seed):
    # Reconstruct the selected estimator without carrying a fitted search object.
    if model_name == "char_tfidf_logistic":
        return LogisticRegression(C=parameters["C"], class_weight="balanced", max_iter=3000, random_state=seed)
    if model_name == "char_tfidf_linear_svm":
        return LinearSVC(C=parameters["C"], class_weight="balanced", random_state=seed)
    if model_name == "char_tfidf_complement_nb":
        return ComplementNB(alpha=parameters["alpha"])
    raise ValueError(f"Unknown model: {model_name}")


def get_score_yes(pipeline, titles, predictions):
    # Prefer native probabilities, then transform margin scores monotonically.
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        class_names = list(classifier.classes_)
        yes_index = class_names.index("Yes")
        return pipeline.predict_proba(titles)[:, yes_index]

    # A logistic transform places LinearSVC margins on a convenient zero-one scale.
    if hasattr(classifier, "decision_function"):
        decision_values = pipeline.decision_function(titles)
        return 1.0 / (1.0 + np.exp(-decision_values))
    return np.array([1.0 if label == "Yes" else 0.0 for label in predictions])


def compute_metrics(gold_labels, predicted_labels, score_yes):
    # Convert the string target into the positive-class representation used by metrics.
    true_values = (gold_labels == "Yes").astype(int)
    predicted_values = (predicted_labels == "Yes").astype(int)
    tn_value, fp_value, fn_value, tp_value = confusion_matrix(true_values, predicted_values, labels=[0, 1]).ravel()

    # Use the same metrics as the LLM experiment for a direct comparison.
    metrics = {
        "accuracy": float(accuracy_score(true_values, predicted_values)),
        "balanced_accuracy": float(balanced_accuracy_score(true_values, predicted_values)),
        "precision_yes": float(precision_score(true_values, predicted_values, zero_division=0)),
        "recall_yes": float(recall_score(true_values, predicted_values, zero_division=0)),
        "f1_yes": float(f1_score(true_values, predicted_values, zero_division=0)),
        "macro_f1": float(f1_score(true_values, predicted_values, average="macro", zero_division=0)),

        # Add specificity, correlation, ranking quality, and raw confusion counts.
        "specificity": float(tn_value / (tn_value + fp_value)) if tn_value + fp_value else 0.0,
        "mcc": float(matthews_corrcoef(true_values, predicted_values)),
        "auroc": float(roc_auc_score(true_values, score_yes)) if len(np.unique(true_values)) == 2 else float("nan"),

        # Preserve all confusion counts for direct comparison with LLM tables.
        "tp": int(tp_value),
        "fp": int(fp_value),
        "tn": int(tn_value),
        "fn": int(fn_value)
    }
    return metrics


def bootstrap_intervals(gold_labels, predicted_labels, rounds, seed):
    # Skip uncertainty estimation only when explicitly requested by the caller.
    if rounds <= 0:
        return {}
    random_generator = np.random.default_rng(seed)
    accuracy_values = []
    macro_f1_values = []

    # Paired resampling preserves each title's gold-prediction relationship.
    for round_index in range(rounds):
        sampled_indices = random_generator.integers(0, len(gold_labels), len(gold_labels))
        sampled_gold = gold_labels.iloc[sampled_indices]
        sampled_predictions = predicted_labels.iloc[sampled_indices]

        # Collect accuracy and macro F1 from the same resampled title indices.
        accuracy_values.append(accuracy_score(sampled_gold, sampled_predictions))
        macro_f1_values.append(f1_score(sampled_gold, sampled_predictions, average="macro", zero_division=0))
    return {
        "accuracy_ci_low": float(np.percentile(accuracy_values, 2.5)),
        "accuracy_ci_high": float(np.percentile(accuracy_values, 97.5)),
        "macro_f1_ci_low": float(np.percentile(macro_f1_values, 2.5)),
        "macro_f1_ci_high": float(np.percentile(macro_f1_values, 97.5))
    }


def extract_top_features(model_name, pipeline, feature_count=30):
    # Read the fitted vocabulary and class-separating feature weights.
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    if model_name == "char_tfidf_complement_nb":
        weights = classifier.feature_log_prob_[list(classifier.classes_).index("Yes")] - classifier.feature_log_prob_[list(classifier.classes_).index("No")]
    else:
        weights = classifier.coef_[0]

    # Positive and negative character n-grams make each linear baseline auditable.
    positive_indices = np.argsort(weights)[-feature_count:][::-1]
    negative_indices = np.argsort(weights)[:feature_count]
    rows = []
    for rank, feature_index in enumerate(positive_indices, start=1):
        rows.append({"model_name": model_name, "direction": "Yes", "rank": rank, "feature": feature_names[feature_index], "weight": float(weights[feature_index])})

    # Store the strongest No-associated features in the same long-form table.
    for rank, feature_index in enumerate(negative_indices, start=1):
        rows.append({"model_name": model_name, "direction": "No", "rank": rank, "feature": feature_names[feature_index], "weight": float(weights[feature_index])})
    return rows


def file_sha256(file_path):
    # Fingerprint every fixed split used by the supervised baseline experiment.
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_run_signature(args):
    # Bind reusable outputs to data content and all modeling settings.
    signature_data = {
        "train_sha256": file_sha256(args.train_file),
        "dev_sha256": file_sha256(args.dev_file),
        "test_sha256": file_sha256(args.test_file),
        "max_features": args.max_features,
        "bootstrap_rounds": args.bootstrap_rounds,
        "seed": args.seed,
        "save_models": args.save_models
    }
    serialized = json.dumps(signature_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def atomic_write_csv(data_frame, output_path):
    # Replace complete baseline tables atomically after all rows are available.
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    data_frame.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    os.replace(temporary_path, output_path)


def atomic_write_json(value, output_path):
    # Write a completion marker only after every baseline artifact is intact.
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_path, output_path)


def outputs_are_current(output_dir, completion_path, run_signature, force):
    # Reuse only a matching complete run with all required result tables present.
    if force or not completion_path.exists():
        return False
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    required_files = ["ml_dev_selection.csv", "ml_predictions.csv", "ml_metrics.csv", "ml_top_features.csv"]
    return completion.get("run_signature") == run_signature and all((output_dir / name).exists() for name in required_files)


def main():
    # Load the immutable train, development, and test partitions.
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = output_dir / "machine_learning_title_detection_complete.json"
    run_signature = make_run_signature(args)
    if outputs_are_current(output_dir, completion_path, run_signature, args.force):
        print(completion_path.read_text(encoding="utf-8"))
        return
    train_data = load_split(args.train_file)
    dev_data = load_split(args.dev_file)
    test_data = load_split(args.test_file).reset_index(drop=True)

    # Tune on development data, then refit selected models on train plus development.
    best_configs, selection_results = select_hyperparameters(train_data, dev_data, args.max_features, args.seed)
    final_train = pd.concat([train_data, dev_data], ignore_index=True)
    prediction_rows = []
    metric_rows = []
    feature_rows = []

    # Add a deterministic keyword baseline before learned models.
    rule_outputs = [rule_predict(title) for title in test_data["title"]]
    rule_predictions = pd.Series([output[0] for output in rule_outputs])
    rule_scores = np.array([output[1] for output in rule_outputs])
    rule_metrics = compute_metrics(test_data["gold_label"], rule_predictions, rule_scores)
    rule_metrics.update(bootstrap_intervals(test_data["gold_label"], rule_predictions, args.bootstrap_rounds, args.seed))
    rule_metrics.update({"model_name": "rule_keywords", "n": int(len(test_data)), "selected_parameters": "{}"})
    metric_rows.append(rule_metrics)

    # Expand deterministic rule outputs to the common row-level prediction schema.
    for row_index, row in test_data.iterrows():
        prediction_rows.append({"model_name": "rule_keywords", "id": row["id"], "title": row["title"], "gold_label": row["gold_label"], "pred_label": rule_predictions.iloc[row_index], "score_yes": float(rule_scores[row_index])})

    # Fit each selected character TF-IDF model and preserve its predictions.
    for model_name, config in best_configs.items():
        estimator = make_estimator(model_name, config["parameters"], args.seed)
        pipeline = make_pipeline(estimator, args.max_features)
        pipeline.fit(final_train["title"].astype(str), final_train["gold_label"].astype(str))

        # Infer test labels and compatible positive-class scores exactly once.
        predicted_labels = pd.Series(pipeline.predict(test_data["title"].astype(str)))
        score_yes = get_score_yes(pipeline, test_data["title"].astype(str), predicted_labels)
        metrics = compute_metrics(test_data["gold_label"], predicted_labels, score_yes)
        metrics.update(bootstrap_intervals(test_data["gold_label"], predicted_labels, args.bootstrap_rounds, args.seed))
        metrics.update({"model_name": model_name, "n": int(len(test_data)), "selected_parameters": json.dumps(config["parameters"], ensure_ascii=False, sort_keys=True)})
        metric_rows.append(metrics)
        feature_rows.extend(extract_top_features(model_name, pipeline))

        # Preserve predictions and positive-class scores for downstream analyses.
        for row_index, row in test_data.iterrows():
            prediction_rows.append({"model_name": model_name, "id": row["id"], "title": row["title"], "gold_label": row["gold_label"], "pred_label": predicted_labels.iloc[row_index], "score_yes": float(score_yes[row_index])})
        if args.save_models:
            model_path = output_dir / f"ml_{model_name}.joblib"
            temporary_model_path = model_path.with_name(model_path.name + ".tmp")
            joblib.dump(pipeline, temporary_model_path)
            os.replace(temporary_model_path, model_path)

    # Save model selection, test predictions, metrics, and interpretable features.
    atomic_write_csv(selection_results, output_dir / "ml_dev_selection.csv")
    atomic_write_csv(pd.DataFrame(prediction_rows), output_dir / "ml_predictions.csv")
    metrics_data = pd.DataFrame(metric_rows).sort_values("macro_f1", ascending=False)
    atomic_write_csv(metrics_data, output_dir / "ml_metrics.csv")
    atomic_write_csv(pd.DataFrame(feature_rows), output_dir / "ml_top_features.csv")

    # Make repeated scheduled runs a cheap signature check after successful fitting.
    completion = {"status": "complete", "run_signature": run_signature, "test_rows": int(len(test_data)), "models": sorted(metrics_data["model_name"].tolist())}
    atomic_write_json(completion, completion_path)
    print(metrics_data.to_string(index=False))


if __name__ == "__main__":
    main()
