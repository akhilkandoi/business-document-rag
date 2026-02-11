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
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("document_classifier")


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

with mlflow.start_run(run_name="LogisticRegression"):

    mlflow.log_param("model_type","LogisticRegression")
    mlflow.log_param("max_iter",1000)
    mlflow.log_param("C",1.0)
    mlflow.log_param("max_features",2500)
    mlflow.log_param("ngram_range","(1,2)")

    #training
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_train_vec, y_train)
    lr_pred=lr.predict(X_test_vec)

    #metrics
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_prec = precision_score(y_test, lr_pred, average='macro')
    lr_recall = recall_score(y_test, lr_pred, average='macro')
    lr_f1 = f1_score(y_test, lr_pred, average='macro')
    lr_train_acc=accuracy_score(y_train, lr.predict(X_train_vec))

    #logging metrics
    mlflow.log_metric("test_accuracy", lr_acc)
    mlflow.log_metric("test_precision", lr_prec)
    mlflow.log_metric("test_recall", lr_recall)
    mlflow.log_metric("test_f1", lr_f1)
    mlflow.log_metric("train_accuracy", lr_train_acc)

    #logging model
    signature = infer_signature(X_train_vec, lr.predict(X_train_vec))
    mlflow.sklearn.log_model(lr, "model", signature=signature)

    #save locally
    models["LogisticRegression"] = (lr, lr_acc, lr_prec, lr_recall, lr_f1)

    print(f"Accuracy: {lr_acc:.2%}\n")
    print(f"Precision: {lr_prec:.2%}\n")
    print(f"Recall: {lr_recall:.2%}\n")
    print(f"F1 score: {lr_f1:.2%}\n")
    print(f"Training accuracy:{lr_train_acc:.2%}")
    print(classification_report(y_test, lr_pred))
    print("\n")



print("===============Random Forest Classifier==============\n")

with mlflow.start_run(run_name="RandomForest"):

    mlflow.log_param("model_type","RandomForest")
    mlflow.log_param("n_estimators",100)
    mlflow.log_param("max_features",2500)
    mlflow.log_param("ngram_range","(1,2)")


    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_vec, y_train)
    rf_pred=rf.predict(X_test_vec)

    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred, average='macro')
    rf_recall = recall_score(y_test, rf_pred, average='macro')
    rf_f1 = f1_score(y_test, rf_pred, average='macro')
    rf_train_acc=accuracy_score(y_train, rf.predict(X_train_vec))

    mlflow.log_metric("test_accuracy", rf_acc)
    mlflow.log_metric("test_precision", rf_prec)
    mlflow.log_metric("test_recall", rf_recall)
    mlflow.log_metric("test_f1", rf_f1)
    mlflow.log_metric("train_accuracy", rf_train_acc)

    signature=infer_signature(X_test_vec, rf.predict(X_train_vec))
    mlflow.sklearn.log_model(rf, "model", signature=signature)

    models["RandomForestClassifier"] = (rf, rf_acc, rf_prec, rf_recall, rf_f1)

    print(f"Accuracy: {rf_acc:.2%}\n")
    print(f"Precision: {rf_prec:.2%}\n")
    print(f"Recall: {rf_recall:.2%}\n")
    print(f"F1 score: {rf_f1:.2%}\n")
    print(f"Training accuracy:{rf_train_acc:.2%}")
    print(classification_report(y_test, rf_pred))
    print("\n")



print("===============XGBoost============\n")

with mlflow.start_run(run_name="XGBoost"):

    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators",100)
    mlflow.log_param("max_features",2500)
    mlflow.log_param("ngram_range","(1,2)")


    xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    xgb.fit(X_train_vec, y_train_encoded)
    xgb_pred = xgb.predict(X_test_vec)
    xgb_pred_labels = label_enc.inverse_transform(xgb_pred)


    xgb_acc = accuracy_score(y_test, xgb_pred_labels)
    xgb_prec = precision_score(y_test, xgb_pred_labels, average='macro')
    xgb_recall = recall_score(y_test, xgb_pred_labels, average='macro')
    xgb_f1 = f1_score(y_test, xgb_pred_labels, average='macro')
    xgb_train_acc=accuracy_score(y_train_encoded, xgb.predict(X_train_vec))

    mlflow.log_metric("test_accuracy", xgb_acc)
    mlflow.log_metric("test_precison", xgb_prec)
    mlflow.log_metric("test_recall", xgb_recall)
    mlflow.log_metric("test_f1", xgb_f1)
    mlflow.log_metric("train_accuracy", xgb_train_acc)

    signature=infer_signature(X_train_vec, xgb.predict(X_train_vec))
    mlflow.sklearn.log_model(xgb, "model", signature=signature)

    models["XGBoost"] = (xgb, xgb_acc, xgb_prec, xgb_recall, xgb_f1)

    print(f"Accuracy: {xgb_acc:.2%}\n")
    print(f"Precision: {xgb_prec:.2%}\n")
    print(f"Recall: {xgb_recall:.2%}\n")
    print(f"F1 score: {xgb_f1:.2%}\n")
    print(f"Training accuracy:{xgb_train_acc:.2%}")
    print(classification_report(y_test, xgb_pred_labels))
    print("\n")


print("======ALL MODEL ACCURACY==========")
for name, (model, acc, prec, rec, f1) in models.items():
    print(f"\n{name}:")
    print(f"  Accuracy:  {acc:.2%}")
    print(f"  Precision: {prec:.2%}")
    print(f"  Recall:    {rec:.2%}")
    print(f"  F1:        {f1:.2%}")


best_name = max(models.items(), key=lambda x:x[1][2])[0] #precision
best_model, best_acc, best_prec, best_rec, best_f1 = models[best_name]
print(f"\n{'='*50}")
print(f"BEST MODEL (by Precision): {best_name}")
print(f"  Precision: {best_prec:.2%} (Primary metric)")
print(f"  Accuracy:  {best_acc:.2%}")
print(f"  Recall:    {best_rec:.2%}")
print(f"  F1:        {best_f1:.2%}")
print(f"{'='*50}\n")

#saving model
os.makedirs("./models", exist_ok=True)
joblib.dump(best_model, "./models/document_classifier.pkl")
joblib.dump(vectorizer, "./models/tfidf_vectorizer.pkl")


metadata={
    "model_type": best_name,
    "selection_criterion": "precision",
    "precision": float(best_prec),
    "accuracy": float(best_acc),
    "recall": float(best_rec),
    "f1_score": float(best_f1),
    "num_features": X_train_vec.shape[1],
    "categories": df['label'].unique().tolist(),
    "training_samples": len(df)
}

with open("./models/classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Models saved in './models' directory locally and logged to MLflow!")
print("Run 'mlflow ui' to view experiments")