import streamlit as st
import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------------------------------------
# CLEAN TEXT FUNCTION
# ------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text.strip()

# ------------------------------------------
# LOAD DATA
# ------------------------------------------
df = pd.read_csv(r"C:\Users\kushw\Downloads\archive\Tweets.csv")

df['clean_text'] = df['text'].apply(clean_text)

# ------------------------------------------
# TF-IDF + MODEL
# ------------------------------------------
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['airline_sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ------------------------------------------
# STREAMLIT UI
# ------------------------------------------
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 25px;
        box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
    }
    .title {
        text-align: center;
        color: #1f77b4;
        font-family: 'Arial', sans-serif;
    }
    .footer {
        text-align: center;
        font-size: 13px;
        color: grey;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>💬 Sentiment Analysis Project</h1>", unsafe_allow_html=True)
st.write("This application analyzes user text and predicts whether the sentiment is **Positive** or **Negative**.")

st.info(f"🎯 Model Accuracy: {acc*100:.2f}%")

user_input = st.text_area("✍️ Enter your sentence here:")

if st.button("Analyze Sentiment"):
    clean_input = clean_text(user_input)
    vectorized = vectorizer.transform([clean_input])
    prediction = model.predict(vectorized)[0]

    if prediction == "positive":
        st.success("✅ Positive Sentiment Detected! 😊")
    elif prediction == "neutral":
        st.info("😐 Neutral Sentiment Detected.")    
    else:
        st.error("❌ Negative Sentiment Detected 😞")

st.markdown("<p class='footer'>Developed by [Saurabh Kushwaha] | Minor Project | AI & ML</p>", unsafe_allow_html=True)