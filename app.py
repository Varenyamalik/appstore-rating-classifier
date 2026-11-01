import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

st.set_page_config(page_title="App Store Rating Predictor", layout="wide")
st.title("Will this App Store app be highly rated?")
st.caption("Upload test data, pick a model, and see how it performs. Models are pre-trained and are not retrained here.")

MODELS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbors": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}
TARGET = "highly_rated"

@st.cache_resource
def load_model(key):
    return joblib.load(f"model/{key}.joblib")

def score(pipe, X, y):
    pred = pipe.predict(X)
    prob = pipe.predict_proba(X)[:, 1]
    return pred, {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, prob),
        "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
    }

file = st.sidebar.file_uploader("Upload test CSV", type="csv")
choice = st.selectbox("Select model", list(MODELS))

if file is None:
    st.info("Upload test_data.csv from the sidebar to begin.")
    st.stop()

data = pd.read_csv(file)
if TARGET not in data.columns:
    st.error(f"The CSV must contain the target column '{TARGET}'.")
    st.stop()

y_true = data[TARGET]
X = data.drop(columns=[TARGET])

with st.expander("Preview uploaded data"):
    st.dataframe(data.head())

pipe = load_model(MODELS[choice])
pred, metrics = score(pipe, X, y_true)

st.subheader(f"Results: {choice}")
cols = st.columns(6)
for col, (name, val) in zip(cols, metrics.items()):
    col.metric(name, f"{val:.3f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y_true, pred)).plot(ax=ax, colorbar=False)
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    st.code(classification_report(y_true, pred))

st.subheader("All models compared on this data")
rows = {}
for label, key in MODELS.items():
    _, m = score(load_model(key), X, y_true)
    rows[label] = m
st.dataframe(pd.DataFrame(rows).T.round(3))