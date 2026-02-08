import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, f1_score, precision_score
import joblib
import os
import json


df = pd.read_csv("./data/processed/question_training_data.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

vectorizer = TfidfVectorizer(
    max_features=2500,
    ngram_range = (1,2),
    stop_words='english'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

label_enc = LabelEncoder()
y_train_encoded = label_enc.fit_transform(y_train)
y_test_encoded = label_enc.transform(y_test)

models={}

print("======Logistic Regression======\n")
lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr.fit(X_train_vec, y_train)
lr_pred=lr.predict(X_test_vec)
lr_acc = accuracy_score(y_test, lr_pred)
lr_prec = precision_score(y_test, lr_pred, average='macro')
lr_recall = recall_score(y_test, lr_pred, average='macro')
lr_f1 = f1_score(y_test, lr_pred, average='macro')
models["LogisticRegression"] = (lr, lr_acc)
print(f"Accuracy: {lr_acc:.2%}\n")
print(f"Precision: {lr_prec:.2%}\n")
print(f"Redcall: {lr_recall:.2%}\n")
print(f"F1 score: {lr_f1:.2%}\n")
print("Classification report:")
print(classification_report(y_test, lr_pred))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, lr_pred))
tc=accuracy_score(y_train, lr.predict(X_train_vec))
print(f"\ntraining accuracy:{tc}")

print("===============Random Forest Classifier==============\n")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_vec, y_train)
rf_pred=rf.predict(X_test_vec)
rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred, average='macro')
rf_recall = recall_score(y_test, rf_pred, average='macro')
rf_f1 = f1_score(y_test, rf_pred, average='macro')
models["RandomForestClassifier"] = (rf, rf_acc)
print(f"Accuracy: {rf_acc:.2%}\n")
print(f"Precision: {rf_prec:.2%}\n")
print(f"Redcall: {rf_recall:.2%}\n")
print(f"F1 score: {rf_f1:.2%}\n")
print("Classification report:")
print(classification_report(y_test, rf_pred))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, rf_pred))
rtc=accuracy_score(y_train, rf.predict(X_train_vec))
print(f"training accuracy:{rtc}")

print("===============XGBoost============\n")

xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train_vec, y_train_encoded)
xgb_pred = xgb.predict(X_test_vec)
xgb_pred_labels = label_enc.inverse_transform(xgb_pred)
xgb_acc = accuracy_score(y_test, xgb_pred_labels)
xgb_prec = precision_score(y_test, xgb_pred_labels, average='macro')
xgb_recall = recall_score(y_test, xgb_pred_labels, average='macro')
xgb_f1 = f1_score(y_test, xgb_pred_labels, average='macro')
models["XGBoost"] = (xgb, xgb_acc)
print(f"Accuracy: {xgb_acc:.2%}\n")
print(f"Precision: {xgb_prec:.2%}\n")
print(f"Redcall: {xgb_recall:.2%}\n")
print(f"F1 score: {xgb_f1:.2%}\n")
print("Classification report:")
print(classification_report(y_test, xgb_pred_labels))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, xgb_pred_labels))
xtc=accuracy_score(y_train_encoded, xgb.predict(X_train_vec))
print(f"training accuracy:{xtc}")


print("======ALL MODEL ACCURACY==========")
for name, (model, acc) in models.items():
    print(f"\n{name}: {acc}")


best_name = max(models.items(), key=lambda x:x[1][1])[0]
best_model, best_acc = models[best_name]
print(f"Best model: {best_name} ({best_acc:.2%})")

#saving model
os.makedirs("./models", exist_ok=True)

joblib.dump(best_model, "./models/document_classifier.pkl")
joblib.dump(vectorizer, "./models/tfidf_vectorizer.pkl")


metadata={
    "model_type":best_name,
    "accuracy":float(best_acc),
    "num_features":X_train_vec.shape[1],
    "categories": df['label'].unique().tolist(),
    "training_samples":len(df)
}

with open("./models/classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Models saved in './models' directory!")