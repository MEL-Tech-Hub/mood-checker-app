import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample training data
TEXT = [
    "I AM JUMPING TODAY",
    "i AM SKIPPY TODAY",
    "i HAVE EATEN TOO MUCH",
    "i AM TOO TIRED",
]
LABELS = ["HAPPY", "ANXIOUS", "FULL", "TIRED"]

# Train model
Vectorizer = CountVectorizer()
x = Vectorizer.fit_transform(TEXT)
model = MultinomialNB()
model.fit(x, LABELS)

# Streamlit UI
st.title("😊 Mood Checker")
user_input = st.text_input("How do you feel today?")

if user_input:  # ✅ check that it's not empty
    user_text = Vectorizer.transform([user_input])
    prediction = model.predict(user_text)
    st.success(f"🧠 MEL thinks you're feeling: **{prediction[0]}**")
