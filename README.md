🛡️ PhishShield AI

Intelligent Phishing Email Detection Using Machine Learning

PhishShield AI is an end-to-end machine learning project that classifies email text as Safe or Potential Phishing using Natural Language Processing (NLP).

The project combines an ML classification pipeline, an interactive Streamlit interface, a FastAPI prediction service, and automated API tests.

Features

- Email classification using TF-IDF and Logistic Regression
- Interactive Streamlit dashboard
- Estimated phishing probability display
- REST API built with FastAPI
- Automated API testing with pytest
- Comparison of Logistic Regression, Linear SVM, and Naive Bayes
- Data cleaning, error analysis, and similarity-aware data splitting

Technology Stack

Languages and ML: Python, Pandas, NumPy, Scikit-learn, TF-IDF

Application: Streamlit, FastAPI, Uvicorn

Testing: Pytest, HTTPX

Model persistence: Joblib

Machine Learning Approach

1. Load and inspect the labelled email dataset.
2. Clean missing and duplicate records.
3. Create training, validation, and testing datasets with similarity-aware grouping.
4. Convert email text into TF-IDF features.
5. Compare multiple classification algorithms using validation metrics.
6. Train and save the Logistic Regression model.
7. Evaluate predictions and analyse classification errors.
8. Make predictions available through Streamlit and FastAPI.

Linear SVM achieved the highest validation F1-score in the final comparison. Logistic Regression was retained for the deployed application because it supports probability estimates and integrates with the existing interface and API.

Model Performance

Logistic Regression — revised internal test evaluation

Metric| Result
Accuracy| 97.46%
Precision| 97.39%
Recall| 97.60%
F1-score| 97.50%
Test examples| 907

The similarity-aware splitting process grouped emails with at least 98% character TF-IDF similarity. A subsequent check found no cross-split matches at that threshold.

Important limitations: These results are from an internal evaluation, not an independent real-world phishing benchmark. Some highly similar emails remain below the 98% threshold, and the source dataset contains examples whose phishing labels may overlap with general spam. Probability estimates are not calibrated security-risk scores.

Installation

Clone or download the project, open its folder in a terminal, and create a Python virtual environment:

py -m venv .venv

Install the dependencies:

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Run the Streamlit Dashboard

.\.venv\Scripts\python.exe -m streamlit run app.py

Open "http://localhost:8501" in your browser.

Paste an email into the text box and select Analyse Email to view the model's classification and estimated phishing probability.

Run the FastAPI Backend

In a separate terminal:

.\.venv\Scripts\python.exe -m uvicorn api:app --reload

Interactive API documentation:

"http://127.0.0.1:8000/docs"

The API provides a "POST /predict" endpoint that accepts email text and returns a prediction with supporting model information.

Run Automated Tests

.\.venv\Scripts\python.exe -m pytest tests/test_api.py -v

The current test suite includes seven passing tests covering API endpoints, example predictions, and invalid inputs.

Dataset

Source: "Balanced Phishing Email Detection Subset — Hugging Face" (https://huggingface.co/datasets/jhonrayo99/phishing-email-balanced-6000)

The original subset contains 6,000 labelled email examples, with "0" representing Safe and "1" representing Phishing. Following cleaning, 5,997 records were used in this project.

The dataset is derived from an earlier phishing-email collection and is published under LGPL-3.0. Refer to the source dataset documentation for provenance and licensing requirements.

Project Scope and Safety

PhishShield AI currently classifies email text only. It does not independently inspect URL reputation, detect malicious attachments, or verify sender identity.

This is an educational machine learning and cybersecurity project. Its predictions should not be used as the sole basis for security decisions.

Author

Saakshi Jatav

MSc Computer Science

"GitHub" (https://github.com/SaakshiJatav) | "LinkedIn" (https://www.linkedin.com/in/saakshi-jatav)