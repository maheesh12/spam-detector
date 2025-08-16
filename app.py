import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from googletrans import Translator

# Load dataset (UCI SMS Spam Collection)
@st.cache_resource
def load_data():
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00228/SMSSpamCollection",
                       sep='\t', header=None, names=["label", "message"])
    return data

data = load_data()

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["message"])
y = data["label"]
model = LogisticRegression()
model.fit(X, y)

# App UI
st.title("📧 Multi-Language Spam Detection")

msg = st.text_area("Enter your message:")

if st.button("Predict"):
    translator = Translator()
    translated = translator.translate(msg, dest="en").text
    vec = vectorizer.transform([translated])
    pred = model.predict(vec)[0]
    st.write("Prediction:", "🚨 SPAM" if pred == "spam" else "✅ NOT SPAM")
