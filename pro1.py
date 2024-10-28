#%%
import os
import pdfplumber
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
import string

#%%
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
#%%
nlp = spacy.load('en_core_web_sm')
# %%
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct and not token.is_space]

    cleaned_text = ' '.join(tokens)

    return cleaned_text

# %%
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return preprocess_text(text)
# %%

