import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from deep_translator import GoogleTranslator

st.title("📧 Multi-Language Spam Detection (Logistic Regression)")

# Step 1: Load dataset
@st.cache_resource
def load_data():
    data = pd.read_csv("spam.csv", sep="\t", header=None, names=["label", "message"])
    return data

data = load_data()
st.write("### Sample Data", data.head())

# Step 2: Encode labels
data["label_num"] = data["label"].map({"ham": 0, "spam": 1})

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    data["message"], data["label_num"], test_size=0.2, random_state=42
)

# Step 4: Text Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Step 6: Evaluate Model
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
st.write("### Model Accuracy:", acc)
st.text("Classification Report:\n" + classification_report(y_test, y_pred))

# Step 7: Predict Function with Translation
st.subheader("🔮 Test Your Own Message")
msg = st.text_area("Enter a message (any language):")
if st.button("Predict"):
    translated = GoogleTranslator(source="auto", target="en").translate(msg)
    vec = vectorizer.transform([translated])
    pred = model.predict(vec)[0]
    st.write("Prediction:", "🚨 SPAM" if pred == 1 else "✅ NOT SPAM")
    st.caption(f"(Translated to English: {translated})")
