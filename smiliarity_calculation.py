#%%
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

#%%
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
# %%
def get_embeddings(text):
    # Tokenize and create embeddings with BERT
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def calculate_similarity(resume_text, job_text):
    # Get embeddings for resume and job description
    resume_embedding = get_embeddings(resume_text)
    job_embedding = get_embeddings(job_text)
    
    # Calculate cosine similarity between the two embeddings
    similarity_score = cosine_similarity(resume_embedding.detach().numpy(), job_embedding.detach().numpy())
    return similarity_score[0][0]
# %%
