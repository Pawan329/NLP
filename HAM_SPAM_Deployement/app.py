import pickle
import streamlit as st

#  Load the model from the pickle file (this is what you'd do in your Streamlit app)
with open('models/spam_classifier.pkl', 'rb') as model_file:
    text_clf = pickle.load(model_file)

# Streamlit App
st.title("SPAM or Not SPAM Message Detection 🕵️")

# User input
user_input = st.text_input("Please Enter Your message here...")


# Predict the output
if user_input:
    output = text_clf.predict([user_input])  # Wrap user_input in a list
    if output[0] == 'spam':
        st.write(f"SPAM 🛑")
    else:
        st.write(f"Not SPAM ✅")
