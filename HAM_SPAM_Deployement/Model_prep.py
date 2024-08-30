import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import streamlit as st

# Load the dataset
df = pd.read_csv('smsspamcollection.tsv', sep='\t')

# Splitting data into features and labels
X = df['message']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create a pipeline with TfidfVectorizer and LinearSVC
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])

# Train the model
model = text_clf.fit(X_train, y_train)

# Save the model as a pickle file
with open('spam_classifier.pkl', 'wb') as model_file:
    pickle.dump(text_clf, model_file)

# # Load the model from the pickle file (this is what you'd do in your Streamlit app)
# with open('spam_classifier.pkl', 'rb') as model_file:
#     text_clf = pickle.load(model_file)

# # Streamlit App
# st.title("SPAM or Not SPAM Message Detection 🕵️")

# # User input
# user_input = st.text_input("Please Enter Your message here...")


# # Predict the output
# if user_input:
#     output = text_clf.predict([user_input])  # Wrap user_input in a list
#     if output[0] == 'spam':
#         st.write(f"SPAM 🛑")
#     else:
#         st.write(f"Not SPAM ✅")
