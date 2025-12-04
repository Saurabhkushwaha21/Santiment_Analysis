import streamlit as st
import pandas as pd
import re
import string
import joblib
import nltk
import spacy

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC  
from sklearn.metrics import accuracy_score

# ------------------------------------------
# NLP SETUP
# ------------------------------------------
nltk.download('stopwords')
nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

# ✅ Important grammar words ko remove mat karo
stop_words.discard('not')
stop_words.discard('no')
stop_words.discard('never')
stop_words.discard('nor')

# ------------------------------------------
# ✅ GRAMMAR-AWARE CLEAN TEXT FUNCTION
# ------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    doc = nlp(text)
    final_words = []

    for token in doc:
        word = token.text
        if word not in stop_words:
            lemma = token.lemma_      # ✅ Grammar-aware root word
            final_words.append(lemma)

    return " ".join(final_words)

# ------------------------------------------
# LOAD DATA
# ------------------------------------------
df = pd.read_csv("Tweets.csv")

df['airline_sentiment'] = df['airline_sentiment'].str.lower().str.strip()
df['clean_text'] = df['text'].apply(clean_text)

# ------------------------------------------
# ✅ TF-IDF WITH GRAMMAR N-GRAMS
# ------------------------------------------
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 3),   # ✅ Grammar pattern check
    min_df=3
)

X = vectorizer.fit_transform(df['clean_text'])
y = df['airline_sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------
# ✅ BEST NLP MODEL (LINEAR SVM)
# ------------------------------------------
model = LinearSVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Save model
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(model, "sentiment_model.pkl")

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
st.write("This application analyzes user text and predicts whether the sentiment is **Positive**, **Negative**, or **Neutral**.")
st.info(f"🎯 Model Accuracy: {acc*100:.2f}%")

# ------------------------------------------
# ✅ GRAMMAR-AWARE FINAL PREDICTION FUNCTION
# ------------------------------------------
def final_predict(text):
    clean_input = clean_text(text)
    vectorized = vectorizer.transform([clean_input])
    prediction = model.predict(vectorized)[0]

    text_lower = text.lower()

    # ✅ Strong grammar-based rules
    if "not good" in text_lower or "not happy" in text_lower or "not like" in text_lower:
        return "negative"

    if "love" in text_lower or "loved" in text_lower or "liked" in text_lower:
        return "positive"

    return prediction

# ------------------------------------------
# USER INPUT
# ------------------------------------------
user_input = st.text_area("✍️ Enter your sentence here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze!")
    else:
        prediction = final_predict(user_input)

        if prediction == "positive":
            st.success("✅ Positive Sentiment Detected! 😊")
        elif prediction == "neutral":
            st.info("😐 Neutral Sentiment Detected.")
        else:
            st.error("❌ Negative Sentiment Detected 😞")

st.markdown("<p class='footer'>Developed by [Saurabh Kushwaha] | Minor Project | AI & ML</p>", unsafe_allow_html=True)
