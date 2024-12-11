#%%
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

#%%
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
# %%
def get_embeddings(text):
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)
# %%
def calculate_similarity(resume_text, job_text):
    
    resume_embedding = get_embeddings(resume_text)
    job_embedding = get_embeddings(job_text)
    
    
    similarity_score = cosine_similarity(resume_embedding.detach().numpy(), job_embedding.detach().numpy())
    return similarity_score[0][0]
    similarity_score = cosine_similarity(resume_embedding.detach().numpy(), job_embedding.detach().numpy())
    return similarity_score[0][0]


# %%
def calculate_similarity_with_feedback(resume_text, job_text):
    resume_embedding = get_embeddings(resume_text)
    job_embedding = get_embeddings(job_text)
    
    # Calculate cosine similarity
    similarity_score = cosine_similarity(resume_embedding.detach().numpy(), job_embedding.detach().numpy())[0][0]

    # Provide feedback based on similarity score
    if similarity_score >= 0.85:
        feedback = "Excellent match! This resume strongly aligns with the job requirements."
    elif similarity_score >= 0.7:
        feedback = "Good match! This candidate would be a suitable fit for the job."
    elif similarity_score >= 0.5:
        feedback = "Fair match. The resume has some relevant experience, but may need adjustments."
    elif similarity_score >= 0.3:
        feedback = "Weak match. This resume has limited alignment with the job requirements."
    else:
        feedback = "Poor match. The resume does not meet the job requirements well."
    
    return similarity_score, feedback

#just to add feed back system better point.
#Adjusted NLP system
