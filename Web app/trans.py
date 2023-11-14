from googletrans import Translator
import streamlit as st
import joblib 


translator = Translator()

st.title("Language Translator")

# Get user input
text = st.text_input("Enter a text")

#Load Model and Vectorizer
model1=joblib.load('/workspaces/Language_Detector-Translator/Models/Language Detector/Language_Detection_Model.joblib')
cv1=joblib.load('/workspaces/Language_Detector-Translator/Models/Language Detector/Vectorizer.joblib')
data = cv1.transform([text]).toarray()
output = model1.predict(data)

message= txt(output)
st.write(message)

# Check if the user has entered any text
if text:
    try:
        # Attempt to translate the text
        translation = translator.translate(text, dest='fr')
        st.write(translation.text)
    except Exception as e:
        # Handle translation errors
        st.error(f"Translation failed: {str(e)}")
else:
    # Inform the user to enter text
    st.warning("Please enter some text for translation.")
