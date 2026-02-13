import joblib
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Load model and vectorizer
classifier = joblib.load("./models/document_classifier.pkl")
vectorizer = joblib.load("./models/tfidf_vectorizer.pkl")

with open("./models/classifier_metadata.json", "r") as f:
    metadata = json.load(f)

print("="*80)
print("MODEL INFO")
print("="*80)
print(f"Model Type: {metadata['model_type']}")
print(f"Selection Criterion: {metadata.get('selection_criterion', 'N/A')}")
print()
print("Single Split Metrics:")
print(f"  Precision: {metadata.get('single_split', {}).get('precision', 'N/A'):.2%}")
print(f"  Accuracy:  {metadata.get('single_split', {}).get('accuracy', 'N/A'):.2%}")
print()

cv = metadata['cross_validation']
print("Cross-Validation Metrics:")
print(f"  CV Precision: {cv.get('cv_precision_mean', 'N/A'):.2%} (±{cv.get('cv_precision_std', 0):.2%})")
print(f"  CV Accuracy:  {cv.get('cv_accuracy', 'N/A'):.2%}")
print(f"  Overfitting Gap: {cv.get('overfitting_gap', cv.get('train_test_gap', 'N/A')):.2%}")
print()
print(f"Categories: {', '.join(metadata['categories'])}")
print("="*80)
print()

# Sample predictions
print("="*80)
print("SAMPLE PREDICTIONS")
print("="*80)

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
        status = "Correct" if confidence > 0.5 else "Wrong"
        print(f"{status} Q: {question}")
        print(f"   → {category} ({confidence:.1%} confidence)\n")
    else:
        print(f"Q: {question}")
        print(f"   → {category}\n")

# Evaluation with test cases
print("\n"+"="*80)
print("DETAILED EVALUATION")
print("="*80)

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
    ("What's the budget approval process?", "FINANCE"),
    
    # GENERAL
    ("What do I do on my first day?", "GENERAL"),
    ("Where is the office located?", "GENERAL"),
    ("What's the WiFi password?", "GENERAL"),
    ("How do I get a parking pass?", "GENERAL"),
]

questions = [q for q, _ in test_cases]
true_labels = [label for _, label in test_cases]

# Vectorize all questions
que_vecs = vectorizer.transform(questions)
pred_labels = classifier.predict(que_vecs)

# Get confidence scores if available
confidences = []
if hasattr(classifier, 'predict_proba'):
    proba_matrix = classifier.predict_proba(que_vecs)
    confidences = [max(proba) for proba in proba_matrix]

# Print individual results
print("\nIndividual Predictions:")
print("-" * 80)
for i, (question, true_label) in enumerate(test_cases):
    pred_label = pred_labels[i]
    is_correct = pred_label == true_label
    
    if confidences:
        conf = confidences[i]
        status = "Right" if is_correct else "Wrong"
        conf_status = "HIGH" if conf > 0.7 else ("MED" if conf > 0.5 else "LOW")
        print(f"{status} [{conf_status} {conf:.1%}] {question}")
        print(f"   Expected: {true_label} | Got: {pred_label}")
    else:
        status = "Right" if is_correct else "Wrong"
        print(f"{status} {question}")
        print(f"   Expected: {true_label} | Got: {pred_label}")
    print()

# Calculate metrics (matching training script)
print("="*80)
print("METRICS (macro-averaged, matching training)")
print("="*80)

accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average='macro')
recall = recall_score(true_labels, pred_labels, average='macro')
f1 = f1_score(true_labels, pred_labels, average='macro')

print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%} (Primary metric)")
print(f"Recall:    {recall:.2%}")
print(f"F1 Score:  {f1:.2%}")
print()

# Detailed classification report
print("Classification Report:")
print("-" * 80)
print(classification_report(true_labels, pred_labels))

# Confusion matrix
print("\nConfusion Matrix:")
print("-" * 80)
cm = confusion_matrix(true_labels, pred_labels, labels=metadata['categories'])
print(f"{'':28s}", end="")
for cat in metadata['categories']:
    print(f"{cat:15s}", end="")
print()
for i, cat in enumerate(metadata['categories']):
    print(f"{cat:15s}", end="")
    for j in range(len(metadata['categories'])):
        print(f"{cm[i][j]:15d}", end="")
    print()

print("\n" + "="*80)
print(f"Overall Test Accuracy: {accuracy:.1%}")
print(f"Correct: {int(accuracy * len(test_cases))}/{len(test_cases)}")
print("="*80)