# Twitter Sentiment & Hate Speech Detection

## Project Overview:
Social media platforms like Twitter generate millions of posts every day, and not all of it is civil. Detecting hate speech and understanding the sentiment behind text at scale is a genuinely hard problem — sarcasm, slang, and context make it tricky even for humans. This project builds a machine learning pipeline that classifies tweets by sentiment (positive, negative, or neutral) and flags hate speech or offensive language, wrapped in a Streamlit web app for real-time predictions.

## Dataset
The dataset is sourced from public Twitter hate speech and sentiment analysis collections. Each record contains the raw tweet text along with two labels — sentiment (positive, negative, neutral) and hate speech classification (hate/offensive or neutral).

## Models
Several classifiers were trained and benchmarked against each other:

Logistic Regression — a strong linear baseline for text classification
Support Vector Machine (SVM) — effective in high-dimensional TF-IDF feature spaces
Random Forest — ensemble of decision trees using bootstrap aggregation
CatBoost — gradient boosting with built-in handling of categorical features
Naive Bayes — probabilistic classifier, fast and well-suited for text
Bagging Classifier — reduces variance by averaging predictions across multiple base learners
Stacking Model — the best performing model, combines predictions from multiple classifiers through a meta-learner

Text was vectorized using TF-IDF before being passed into any of the models, converting raw tweet text into numerical feature representations based on term frequency and inverse document frequency.

Web App
The project ships with a Streamlit interface where users can type any text or tweet and get back real-time predictions on both sentiment and hate speech classification. The app also displays a model accuracy comparison so you can see how each classifier performed relative to the others.

## To Lauch this :
--> just run this app.py in any of your VS code and type : "streamlit run app.py" in the terminal 
--> That will bring you to the streamlit interface where you can directly type and check if the model is working correctly or not.

## Prepare TweetClaw Exports

Use `scripts/predict_tweetclaw_export.py` to convert reviewed TweetClaw JSON,
JSONL, or CSV exports into a local CSV for batch review. When `stacking_model.pkl`
and `tfidf_vectorizer.pkl` are available, the script also adds the same hate
speech prediction used by the Streamlit app.

```bash
python scripts/predict_tweetclaw_export.py examples/tweetclaw_export.jsonl \
  --output tweetclaw_predictions.csv
```

Review exports before conversion and do not commit private account data,
credentials, or non-public tweets. TweetClaw is available from
https://github.com/Xquik-dev/tweetclaw.

Built with Python · Scikit-learn · Streamlit · CatBoost · NLTK · Pandas · NumPy
