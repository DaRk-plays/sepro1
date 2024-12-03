from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
from model.model_code import extract_text_from_pdf, calculate_similarity_with_feedback

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def resume_matching():
    return render_template("index.html", title="Resume Matching")

@app.route("/profile")
def profile():
    user_data = {
        "username": "John Doe",
        "resumes_uploaded": 3,
        "matches_reviewed": 12
    }
    return render_template("profile.html", title="Profile", user_data=user_data)

@app.route("/upload", methods=["POST"])
def upload_files():
    if "resume" not in request.files or "job_description" not in request.files:
        return jsonify({"error": "Both resume and job description files are required"}), 400

    resume = request.files["resume"]
    job_description = request.files["job_description"]

    resume_path = os.path.join(UPLOAD_FOLDER, resume.filename)
    job_description_path = os.path.join(UPLOAD_FOLDER, job_description.filename)
    
    resume.save(resume_path)
    job_description.save(job_description_path)

    resume_text = extract_text_from_pdf(resume_path)
    job_text = extract_text_from_pdf(job_description_path)

    similarity, feedback = calculate_similarity_with_feedback(resume_text, job_text)

    return jsonify({"similarity_score": round(similarity, 2), "feedback": feedback})

@app.route("/logout")
def logout():
    return redirect(url_for("resume_matching"))

if __name__ == "__main__":
    app.run(debug=True)

