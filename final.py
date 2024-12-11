#%%
from model import processed_resumes, processed_job_descriptions
# %%
from smiliarity_calculation import calculate_similarity
# %%
for resume_name, resume_text in processed_resumes.items():
    print(f"\nResume: {resume_name}")
    for index, job_row in processed_job_descriptions.iterrows():
        job_text = job_row['processed_description']
        job_title = job_row.get('title', 'Job Title Not Available')
        score = calculate_similarity(resume_text, job_text)
        print(f"Job Title: {job_title} | Matching Score: {score * 100:.2f}%")
# %%
#adjusted NLP system

# %%
