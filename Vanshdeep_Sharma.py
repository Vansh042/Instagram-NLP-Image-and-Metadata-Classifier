
"""
Pav-Bhaji Text Classification
--------------------------------
Text-only classification of Instagram posts using pavbhaji.json metadata.
Ground truth:
    images/1 -> Pav Bhaji
    images/0 -> Not Pav Bhaji

No image pixels, CNNs, pretrained vision models, or image embeddings are used.
"""

import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)


RANDOM_STATE = 42


def clean_text(text):
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^\w#\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def caption_from_record(record):
    try:
        edges = record.get("edge_media_to_caption", {}).get("edges", [])
        if edges:
            return edges[0].get("node", {}).get("text", "") or ""
    except (AttributeError, TypeError):
        pass
    return ""


def url_basename(url):
    if not url:
        return None
    return os.path.basename(urlparse(str(url)).path)


def load_dataset(data_dir):
    data_dir = Path(data_dir)
    json_path = data_dir / "pavbhaji.json"
    images_dir = data_dir / "images"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing {images_dir}")

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Ground-truth label from image folders.
    label_by_filename = {}
    for label in (0, 1):
        folder = images_dir / str(label)
        if not folder.exists():
            raise FileNotFoundError(f"Missing {folder}")
        for p in folder.iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                label_by_filename[p.name] = label

    rows = []
    for idx, record in enumerate(records):
        candidates = [
            url_basename(record.get("display_url")),
            url_basename(record.get("thumbnail_src")),
        ]
        image_name = next((x for x in candidates if x in label_by_filename), None)
        if image_name is None:
            continue

        caption = caption_from_record(record)
        tags = " ".join(record.get("tags") or [])

        rows.append({
            "json_index": idx,
            "image": image_name,
            "label": label_by_filename[image_name],
            "caption": caption,
            "tags": tags,
            "likes": (record.get("edge_media_preview_like") or {}).get("count"),
            "comments": (record.get("edge_media_to_comment") or {}).get("count"),
            "is_video": record.get("is_video"),
            "location": (
                record.get("location", {}).get("name", "")
                if isinstance(record.get("location"), dict) else ""
            ),
            "timestamp": record.get("taken_at_timestamp"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No image-to-JSON matches were found.")

    # Main predictive text feature.
    df["text_raw"] = (df["caption"].fillna("") + " " + df["tags"].fillna("")).str.strip()
    df["text"] = df["text_raw"].map(clean_text)
    df = df[df["text"].str.len() > 0].copy()

    return df, len(records), len(label_by_filename)


def save_eda(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Class distribution
    counts = df["label"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["0: Non-Pav Bhaji", "1: Pav Bhaji"],
            [counts.get(0, 0), counts.get(1, 0)])
    plt.ylabel("Number of samples")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=160)
    plt.close()

    # Text length
    lengths = df["text"].str.split().str.len()
    plt.figure(figsize=(7, 4))
    plt.hist(lengths, bins=20)
    plt.xlabel("Word count")
    plt.ylabel("Number of posts")
    plt.title("Text Length Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "text_length_distribution.png", dpi=160)
    plt.close()

    # Pavbhaji keyword occurrence
    has_term = df["text"].str.contains("pavbhaji", regex=False)
    kw = pd.crosstab(has_term, df["label"]).reindex(index=[False, True], columns=[0, 1], fill_value=0)
    plt.figure(figsize=(6, 4))
    x = np.arange(2)
    width = 0.35
    plt.bar(x - width/2, kw[0].values, width, label="0: Non-Pav Bhaji")
    plt.bar(x + width/2, kw[1].values, width, label="1: Pav Bhaji")
    plt.xticks(x, ["No 'pavbhaji'", "Contains 'pavbhaji'"])
    plt.ylabel("Number of posts")
    plt.title("Pav-Bhaji Keyword Presence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "keyword_presence.png", dpi=160)
    plt.close()


def build_models():
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_df=0.98
    )
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", vectorizer),
            ("classifier", LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
            ))
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True, ngram_range=(1, 2),
                sublinear_tf=True, min_df=1, max_df=0.98
            )),
            ("classifier", LinearSVC(
                C=0.3, class_weight="balanced", random_state=RANDOM_STATE
            ))
        ]),
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True, ngram_range=(1, 2),
                min_df=1, max_df=0.98
            )),
            ("classifier", MultinomialNB())
        ])
    }


def evaluate_holdout(models, X_train, X_test, y_train, y_test):
    rows = []
    fitted = {}
    matrices = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
        })
        fitted[name] = model
        matrices[name] = confusion_matrix(y_test, pred)

    return pd.DataFrame(rows), fitted, matrices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="dataset",
        help="Path containing pavbhaji.json and images/0, images/1"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for CSVs and figures"
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df, json_count, image_count = load_dataset(args.data_dir)

    print("=" * 70)
    print("PAV-BHAJI TEXT CLASSIFICATION")
    print("=" * 70)
    print(f"JSON metadata records: {json_count}")
    print(f"Images with labels:     {image_count}")
    print(f"Matched training rows:  {len(df)}")
    print("\nClass distribution:")
    print(df["label"].value_counts().sort_index())
    print("\nAvailable dataset columns:")
    print(df.columns.tolist())

    print("\nFeature summary:")
    print("Predictive text = caption + hashtags/tags")
    print("Ignored as predictive features = image filename, JSON index, post ID/shortcode")
    print("Metadata-only fields such as likes/comments/location are retained for analysis,")
    print("but are not used in the baseline model to avoid relying on weak/non-semantic signals.")

    # EDA
    save_eda(df, out / "figures")

    # Save cleaned data for inspection.
    df.to_csv(out / "matched_metadata.csv", index=False)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    models = build_models()
    results, fitted, matrices = evaluate_holdout(
        models, X_train, X_test, y_train, y_test
    )

    results.to_csv(out / "holdout_model_results.csv", index=False)
    print("\nHoldout results (positive class = Pav Bhaji):")
    print(results.to_string(index=False))

    # 5-fold stratified cross-validation for a more stable comparison.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for name, model in models.items():
        scores = cross_validate(
            model, X, y, cv=cv,
            scoring=["accuracy", "precision", "recall", "f1"],
            n_jobs=None
        )
        cv_rows.append({
            "Model": name,
            "CV Accuracy Mean": scores["test_accuracy"].mean(),
            "CV Precision Mean": scores["test_precision"].mean(),
            "CV Recall Mean": scores["test_recall"].mean(),
            "CV F1 Mean": scores["test_f1"].mean(),
        })
    cv_results = pd.DataFrame(cv_rows)
    cv_results.to_csv(out / "cross_validation_results.csv", index=False)

    print("\n5-fold stratified CV results:")
    print(cv_results.to_string(index=False))

    # Confusion matrix for Logistic Regression (final example model).
    final_name = "Logistic Regression"
    cm = matrices[final_name]
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Pav Bhaji", "Pav Bhaji"]
    ).plot(ax=ax, values_format="d")
    ax.set_title("Linear SVM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out / "figures" / "linear_svm_confusion_matrix.png", dpi=160)
    plt.close()

    print("\nLinear SVM classification report:")
    print(classification_report(
        y_test, fitted[final_name].predict(X_test),
        target_names=["Non-Pav Bhaji", "Pav Bhaji"],
        zero_division=0
    ))

    # Train final model on all available matched rows.
    final_model = models[final_name]
    final_model.fit(X, y)

    # Save a few example predictions.
    sample_text = [
        "delicious pav bhaji with butter pav",
        "chicken tikka with mint chutney",
        "street food and masala bhaji"
    ]
    sample_pred = final_model.predict(sample_text)
    pd.DataFrame({
        "text": sample_text,
        "prediction": sample_pred,
        "prediction_label": np.where(sample_pred == 1, "Pav Bhaji", "Not Pav Bhaji")
    }).to_csv(out / "example_predictions.csv", index=False)

    print("\nExample predictions:")
    for text, pred in zip(sample_text, sample_pred):
        print(f"{pred}: {text}")

    print(f"\nResults saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
