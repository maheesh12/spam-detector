import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from deep_translator import GoogleTranslator

# Load dataset
@st.cache_resource
def load_data():
    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/SMSSpamCollection",
        sep='\t',
        header=None,
        names=["label", "message"]
    )
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
    # Translate any language to English
    translated = GoogleTranslator(source='auto', target='en').translate(msg)
    
    # Predict
    vec = vectorizer.transform([translated])
    pred = model.predict(vec)[0]
    st.write("Prediction:", "🚨 SPAM" if pred == "spam" else "✅ NOT SPAM")

