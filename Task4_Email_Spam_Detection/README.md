# Task 4 — Email Spam Detection with Machine Learning

An end-to-end NLP classifier that identifies spam messages and exposes
probability-based predictions through a polished Streamlit interface.

## Method

1. Remove unused columns and duplicate messages.
2. Create a stratified train/test split.
3. Transform text with TF-IDF unigrams and bigrams.
4. Train class-balanced logistic regression.
5. Evaluate accuracy, spam precision, recall, and the confusion matrix.

## Run

```bash
python train_model.py
streamlit run app.py
```

The trained pipeline contains both text preprocessing and the classifier, which
prevents inconsistent preprocessing during prediction.

