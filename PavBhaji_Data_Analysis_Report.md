# Pav-Bhaji Text Classification — Data Analysis Report

## 1. Problem Statement

The objective is to classify whether an Instagram post/image belongs to the Pav-Bhaji class or the non-Pav-Bhaji class.

This is explicitly a **text-based classification problem**. The images are not processed by a CNN or any other computer-vision model. The image folders provide the ground-truth labels and allow each supplied image filename to be linked to its Instagram metadata record in `pavbhaji.json`.

**Label mapping**
- `1` = Pav Bhaji
- `0` = Not Pav Bhaji

## 2. Dataset Inspection

The supplied archive contains:
- `pavbhaji.json`: 1,500 Instagram metadata records.
- `images/0`: 269 labelled non-Pav-Bhaji images.
- `images/1`: 183 labelled Pav-Bhaji images.
- Total supplied labelled images: 452.

The filenames of the supplied images were matched to the basename of `display_url` (or `thumbnail_src`) in the JSON. This produced **452 usable image–metadata matches**, which form the supervised-learning dataset.

There are 1,048 JSON records without a corresponding supplied labelled image, so they are not used for supervised training.

## 3. JSON Features and Relevance

| Feature | Type | Use | Relevance |
|---|---|---|---|
| `edge_media_to_caption` | Nested text | Yes | High: contains post description/caption text. |
| `tags` | List of text | Yes | High: hashtags are strong lexical/contextual signals. |
| `location.name` | Categorical/text | No in baseline | Low–Medium: contextual but not a direct food-content signal. |
| `edge_media_preview_like.count` | Numeric | No | Low: engagement is not semantic food information. |
| `edge_media_to_comment.count` | Numeric | No | Low: engagement is not semantic food information. |
| `is_video` | Boolean | No | Low: media type does not identify the food. |
| `taken_at_timestamp` | Numeric/date | No | Low: not a direct food-content signal. |
| `video_view_count` | Numeric | No | Low and highly missing (only 110/1,500 records have it). |
| `location` | Nested object | No in baseline | Low–Medium; often missing. |
| `id`, `shortcode` | Identifier | No | None: unique IDs should not become predictive features. |
| `display_url`, `thumbnail_src` | URL | Used for matching only | None as model input. |
| `image` filename | Identifier | Used for label matching only | None as model input. |
| `dimensions` | Nested numeric | No | Low: image size is not semantic text information. |
| `owner` | Nested identifier | No | None/low; would risk learning account-specific artifacts. |
| `thumbnail_resources`, `urls` | Lists/URLs | No | Not needed for text classification. |
| `comments_disabled` | Boolean | No | Not a food-content feature. |

### Actual JSON structure

The top-level fields found in the supplied JSON are:

`comments_disabled`, `dimensions`, `display_url`, `edge_liked_by`, `edge_media_preview_like`, `edge_media_to_caption`, `edge_media_to_comment`, `id`, `is_video`, `location`, `owner`, `shortcode`, `tags`, `taken_at_timestamp`, `thumbnail_resources`, `thumbnail_src`, `urls`, `video_view_count`.

## 4. Data Cleaning

The model uses the concatenation:

**caption/description + tags/hashtags**

Preprocessing includes:
1. Convert text to lowercase.
2. Remove URLs.
3. Remove Instagram `@mentions`.
4. Remove unnecessary punctuation/symbols.
5. Normalize whitespace.
6. Retain words and hashtag text.
7. Remove rows with empty final text.

The combined text has an average length of approximately **68 words** per matched post, with a median of **67 words**.

There were **7 matched rows without tags** and **11 duplicate combined-text rows**. Missing tags are handled as empty text.

## 5. Important Dataset Observation

The dataset description states that the source data contains posts associated with `#pavbhaji` in their description/comment. In the 452 matched rows, the string `pavbhaji` occurs in **443 rows (98.0%)**.

Therefore, the classifier is **not simply detecting whether the word "pavbhaji" exists**. Since that word is present in almost every matched sample, the model must use other textual patterns, word combinations, and contextual terms to distinguish the image-derived classes.

This also creates a selection bias: the data is not a random sample of all Instagram food posts. The reported results should therefore be interpreted as performance on this supplied dataset, not as general image-recognition accuracy.

## 6. Exploratory Data Analysis

### Class distribution

The supervised dataset contains:
- Non-Pav Bhaji (`0`): 269 (59.5%)
- Pav Bhaji (`1`): 183 (40.5%)

This is a moderate class imbalance, so the experiments include class-weight balancing for Logistic Regression and Linear SVM.

### Text features

The main predictive representation is **TF-IDF** with word unigrams and bigrams (`ngram_range=(1,2)`). TF-IDF gives higher weight to terms that are informative within the corpus while reducing the effect of very common terms.

## 7. Model Training

Three classical text-classification algorithms were compared:

1. Logistic Regression with `class_weight="balanced"`.
2. Linear SVM with `C=0.3` and `class_weight="balanced"`.
3. Multinomial Naive Bayes.

A stratified 80/20 train-test split with `random_state=42` was used for the holdout evaluation. A separate 5-fold stratified cross-validation was also performed.

No image pixels or computer-vision features are used.

## 8. Holdout Evaluation

Positive class = **Pav Bhaji (`1`)**.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.626 | 0.524 | 0.892 | 0.660 |
| Linear SVM | 0.615 | 0.516 | 0.865 | 0.646 |
| Multinomial Naive Bayes | 0.593 | 0.500 | 0.108 | 0.178 |

The Logistic Regression holdout results are:
- Accuracy: **0.626**
- Precision: **0.524**
- Recall: **0.892**
- F1-score: **0.660**

Its holdout confusion matrix is:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 22 | 32 |
| Actual 1 | 2 | 35 |

Thus, for the holdout set, the Logistic Regression model correctly classified 22 non-Pav-Bhaji posts and 35 Pav-Bhaji posts. It produced 32 false positives and 2 false negatives.

## 9. Five-Fold Cross-Validation

| Model | Mean Accuracy | Mean Precision | Mean Recall | Mean F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.684 | 0.579 | 0.814 | 0.676 |
| Linear SVM | 0.692 | 0.594 | 0.776 | 0.672 |
| Multinomial Naive Bayes | 0.604 | 0.573 | 0.077 | 0.134 |

Cross-validation gives a more stable estimate than a single train/test split. The Logistic Regression and Linear SVM models have similar F1 performance, while Multinomial Naive Bayes has substantially lower recall and F1 on this dataset.

## 10. Final Prediction Workflow

For a new Instagram post with available metadata:

```text
Caption + Tags
      ↓
Text cleaning
      ↓
TF-IDF (unigrams + bigrams)
      ↓
Logistic Regression
      ↓
0 = Not Pav Bhaji
1 = Pav Bhaji
```

The model predicts from the **metadata text associated with the image**, not from the image pixels.

## 11. Limitations

1. Only 452 image–metadata matches are available for supervised learning.
2. The source dataset is preselected around the Pav-Bhaji hashtag/topic.
3. `pavbhaji` appears in 443/452 matched texts, limiting the usefulness of simple keyword rules.
4. Instagram captions and hashtags may contain noisy, promotional, copied, or multilingual text.
5. The results do not measure visual recognition performance.
6. A different sample of Instagram posts may produce different performance.
7. The supplied JSON contains 1,500 metadata records, but only 452 can be paired with the labelled images.

## 12. Conclusion

A complete text-classification pipeline was implemented using Instagram metadata, TF-IDF features, and classical machine-learning algorithms. The image folders were used only to establish the supervised labels and to match images to their metadata.

The experiments show that classical text models can learn useful signals from the supplied captions and hashtags, but the dataset's small size and hashtag-based selection mean that the measured performance should be treated as dataset-specific rather than as general Pav-Bhaji image-recognition performance.

## 13. Deliverables

The submission package contains:
- `pavbhaji_classifier.py` — executable source code.
- `pavbhaji_classifier.ipynb` — notebook version.
- `PavBhaji_Data_Analysis_Report.pdf` — formatted report.
- `PavBhaji_Data_Analysis_Report.md` — editable report source.
- `requirements.txt` — Python dependencies.
- `README.md` — execution instructions.
- `results/` — generated CSVs and EDA figures.
