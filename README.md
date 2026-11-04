# Pav-Bhaji Text Classification

## What this project does

Classifies Instagram posts as:

- `1` = Pav Bhaji
- `0` = Not Pav Bhaji

This is a **text-based classification** solution. No CNN or image-pixel features are used.

The supplied image folders provide ground-truth labels and are used to match image filenames to records in `pavbhaji.json`.

## Files

- `pavbhaji_classifier.py` — complete executable source code.
- `pavbhaji_classifier.ipynb` — notebook version.
- `PavBhaji_Data_Analysis_Report.pdf` — submission-ready report.
- `PavBhaji_Data_Analysis_Report.md` — editable report.
- `requirements.txt` — dependencies.
- `results/` — generated EDA figures, matched data and evaluation tables.

## How to run

1. Extract the original `dataset.zip`.
2. Put the extracted `dataset` folder in the same directory as the Python script, or pass its path explicitly.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run:

```bash
python pavbhaji_classifier.py --data-dir dataset --output-dir results
```

The program will inspect the JSON, match image filenames to metadata, preprocess text, train three classifiers, evaluate them on a stratified holdout set, run 5-fold cross-validation, save EDA plots and evaluation CSVs, and show example predictions.

## Important

The JSON contains 1,500 metadata records, but the supplied labelled images allow 452 image–metadata matches. Only these 452 rows are used for supervised learning.

The model uses:
- Instagram caption/description
- Instagram tags/hashtags

It does not use:
- image pixels
- CNNs
- image embeddings
- image filename as a predictive feature
- post ID as a predictive feature

## Viva explanation

The core pipeline is:

`Image folder label -> match image filename with JSON metadata -> caption + tags -> text preprocessing -> TF-IDF -> classifier -> 0/1 prediction`

The image is therefore not visually analyzed. It supplies the ground-truth class for the metadata record.
