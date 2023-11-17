import streamlit as st
import pandas as pd
from googletrans import Translator
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import joblib

# Load the pre-trained language detection model
model1=joblib.load('/workspaces/Language_Detector-Translator/Models/Language Detector/Language_Detection_Model.joblib')
cv1=joblib.load('/workspaces/Language_Detector-Translator/Models/Language Detector/Vectorizer.joblib')

translate=...
#function to detect the language
def detect_language(text):
    text_vectorized = cv1.transform([text]).toarray()
    prediction = model1.predict(text_vectorized)
    return prediction[0]

#function for translation
def translate_language(text):
    translator = Translator()
    text = input("Enter a text")
    translate = translator.translate(text, dest='fr')
    final_text= translate.text
    
    return final_text



# Create the Streamlit web app
def main():
    st.title("Language Detection Web App")
    st.sidebar.header("User Input")

    # Get user input
    user_input = st.sidebar.text_area("Enter Text for Language Detection:")

    if st.sidebar.button("Detect Language"):
        if user_input:
            # Use the language detection function
            language = detect_language(user_input)
            st.success(f"Detected Language: {language}")
        else:
            st.warning("Please Enter Some Text For Language Detection.")

            
    if st.sidebar.button("Translate Text"):
        tanslate= translate_language(user_input)
        st.success(f"Translated to: {translate}")

if __name__ == "__main__":
    main()
