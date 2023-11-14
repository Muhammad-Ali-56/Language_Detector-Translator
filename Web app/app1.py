import streamlit as st
from googletrans import Translator
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Load the pre-trained language detection model
model1 = joblib.load('C:/Users/pc/Desktop/Final/Language_Detector-Translator/Models/Language Detector/Language_Detection_Model.joblib')
cv1 = joblib.load('C:/Users/pc/Desktop/Final/Language_Detector-Translator/Models/Language Detector/Vectorizer.joblib')

# Function to detect the language
def detect_language(text):
    text_vectorized = cv1.transform([text]).toarray()
    prediction = model1.predict(text_vectorized)
    return prediction[0]

# Function for translation
def translate_language(text, target_language):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    final_text = translation.text
    return final_text

# Create the Streamlit web app
def main():
    st.title("Language Detection and Translation Web App")
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

    # Add a selectbox for choosing the target language
    target_language = st.sidebar.selectbox("Select Target Language:", ["fr", "ur", "ar", "la"])
    
    if st.sidebar.button("Translate Text"):
        translation_result = translate_language(user_input, target_language)
        st.success(f"Translated to {target_language.upper()}:  {translation_result}")

if __name__ == "__main__":
    main()
