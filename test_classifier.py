import joblib
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

classifier = joblib.load("./models/document_classifier.pkl")
vectorizer = joblib.load("./models/tfidf_vectorizer.pkl")

with open("./models/classifier_metadata.json", "r") as f:
    metadata = json.load(f)

print("="*80)
print(f"Model: {metadata['model_type']}")
print(f"Training Accuracy: {metadata['accuracy']:.2%}")
print(f"Categories: {', '.join(metadata['categories'])}")
print("="*80)
print()


print("Predictions")
print("=" * 60)

test_questions = [
    "How many vacation days do I get?",
    "What's the parental leave policy?",
    "How do I submit a pull request?",
    "What's the expense approval limit?",
    "What do I do on my first day?",
    "How does 401k matching work?",
    "What's the code review process?",
    "Can I work remotely?",
    "How do I request reimbursement?",
    "What's our tech stack?",
    "How do I set up my development environment?",
    "What are the health insurance options?"
]

for question in test_questions:
    que_vec = vectorizer.transform([question])
    category = classifier.predict(que_vec)[0]

    if hasattr(classifier, 'predict_proba'):
        proba = classifier.predict_proba(que_vec)[0]
        confidence = max(proba)
        print(f"Q: {question}")
        print(f"-> {category} ({confidence:.0%})\n")
    else:
        print(f"Q: {question}")
        print(f"-> {category}\n")


print("\n"+"="*80)
print("EVALUATION")
print("="*80)

#with labels
test_cases = [
    # HR
    ("How many vacation days do I get?", "HR"),
    ("What's the parental leave policy?", "HR"),
    ("Can I work remotely?", "HR"),
    ("What are the health insurance options?", "HR"),
    ("What's the sick leave policy?", "HR"),
    
    # ENGINEERING
    ("How do I submit a pull request?", "ENGINEERING"),
    ("What's the code review process?", "ENGINEERING"),
    ("What's our tech stack?", "ENGINEERING"),
    ("What's the deployment process?", "ENGINEERING"),
    ("How do I set up my development environment?", "ENGINEERING"),
    
    # FINANCE
    ("What's the expense approval limit?", "FINANCE"),
    ("How does 401k matching work?", "FINANCE"),
    ("How do I request reimbursement?", "FINANCE"),
    ("How do I submit an invoice?", "FINANCE"),
    
    # GENERAL
    ("What do I do on my first day?", "GENERAL"),
    ("Where is the office located?", "GENERAL"),
    ("What's the WiFi password?", "GENERAL"),
]

correct = 0
total = len(test_cases)

for question, expected_label in test_cases:
    que_vec = vectorizer.transform([question])
    pred_label = classifier.predict(que_vec)[0]

    if pred_label == expected_label:
        correct += 1
        print(f"Correct. {question}")
        print(f"->{pred_label}\n")
    else:
        print(f"Wrong. {question}")
        print(f" Expected: {expected_label} | Got: {pred_label}\n")


print("="*60)
accuracy = (correct/total)*100
print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")

