import os
import pdfplumber
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
import string

#%%
resume_main_folder = "D:\SE project\data\archive (2)\data\data"

csv_file_path = "D:\SE project\data\archive (1)\postings.csv"
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
