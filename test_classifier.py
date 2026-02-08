import joblib
import json

classifier = joblib.load("./models/document_classifier.pkl")
vectorizer = joblib.load("./models/tfidf_vectorizer.pkl")

with open("./models/classifier_metadata.json","r") as f:
    metadata = json.load(f)

test_questions = [
    "How many vacation days do I get?",
    "What's the parental leave policy?",
    "How do I submit a pull request?",
    "What's the expense approval limit?",
    "What do I do on my first day?",
    "How does 401k matching work?",
    "What's the code review process?",
    "Can I work remotely?",
    "How do I request reimbursement?"
]

print("Prediction test:")

for que in test_questions:

    que_vc = vectorizer.transform([que])

    cat = classifier.predict(que_vc)[0]
    proba = classifier.predict_proba(que_vc)[0]
    confidence = max(proba)

    print(f"Question: {que}")
    print(f"Category: {cat} ({confidence:.1%})")

