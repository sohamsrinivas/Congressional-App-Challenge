# TicketIQ

Support ticket triage for small teams. Paste a message (or upload a CSV of them) and get: department, priority, sentiment, emotion, topic, churn risk, a 0 to 100 business risk score, and concrete next steps.

Built from an internship integration project; repackaged as a public web app for the Congressional App Challenge.

## Run locally

```bash
pip install -r requirements.txt
python -m textblob.download_corpora   # one time
streamlit run app.py
```

## Deploy (free, about 5 minutes)

1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io, sign in with GitHub, click "New app".
3. Pick the repo, branch `main`, main file `app.py`. Deploy.
4. You get a public URL like `https://<name>.streamlit.app`. Put that in the submission.

Vercel does not run Streamlit. If you want a custom domain, host a one-page landing site on Vercel that links to the Streamlit URL.

## Model files

Put trained artifacts in an `artifacts/` folder next to `app.py`:

```
artifacts/
  ticket_classifier_model.pkl
  tfidf_vectorizer.pkl
  business_mapping.pkl
  churn_model.pkl
  scaler.pkl
  feature_columns.pkl
  bertopic_model            (optional, see below)
  business_topic_labels.pkl (optional)
```

If a file is missing the app still runs and uses a clearly labeled rule-based fallback for that stage. The sidebar shows which models are live.

Files over 100 MB need Git LFS, and Streamlit Cloud does not pull LFS files. If the BERTopic model is large, leave it out: the keyword topic tagger covers the demo and the app stays within the free tier's 1 GB memory limit. The sklearn pickles are small and safe to commit.

Pickles must be loaded with the same scikit-learn major version they were saved with. If you get a load error, pin `scikit-learn==<version from your Colab>` in `requirements.txt`.
