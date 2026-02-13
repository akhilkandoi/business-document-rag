import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
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
    mlflow.sklearn.log_model(lr, name="model", signature=signature)

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

    signature=infer_signature(X_train_vec, rf.predict(X_train_vec))
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
    mlflow.log_metric("test_precision", xgb_prec)
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


print("="*80)
print("CROSS VALIDATION ANALYSIS (5-fold)")
print("="*80)


pipelines = {
    'LogisticRegression':Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2500, ngram_range=(1,2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, C=1.0))
    ]),
    'RandomForest':Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2500, ngram_range=(1,2), stop_words='english')),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    'XGBoost':Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2500, ngram_range=(1,2), stop_words='english')),
        ('clf', XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
    ]),
}

X_full = df['text']
y_full = df['label']

cv_results = {}

for name, pipeline in pipelines.items():
    print(f"\n{name}:")

    #for xgboost
    if name == "XGBoost":
        y_cv = label_enc.fit_transform(y_full)
    else:
        y_cv = y_full
    
    cv_scores = cross_validate(
        pipeline,
        X_full,
        y_cv,
        cv=5, 
        scoring = {
            'precision': 'precision_macro',
            'accuracy': 'accuracy',
            'recall': 'recall_macro',
            'f1':'f1_macro'
        },
        return_train_score = True
    )

    #calculate mean ans std
    cv_train_prec  = cv_scores['train_precision'].mean()
    cv_test_prec = cv_scores['test_precision'].mean()
    cv_test_prec_std = cv_scores['test_precision'].std()

    cv_test_acc = cv_scores['test_accuracy'].mean()
    cv_test_rec = cv_scores['test_recall'].mean()
    cv_test_f1 = cv_scores['test_f1'].mean()

    gap = cv_train_prec - cv_test_prec

    cv_results[name]={
        'cv_precision_mean':cv_test_prec,
        'cv_precision_std':cv_test_prec_std,
        'cv_accuracy':cv_test_acc,
        'cv_recall':cv_test_rec,
        'cv_f1':cv_test_f1,
        'train_test_gap':gap,
    }

    print(f"  CV Precision:     {cv_test_prec:.2%} (±{cv_test_prec_std:.2%})")
    print(f"  CV Accuracy:      {cv_test_acc:.2%}")
    print(f"  CV Recall:        {cv_test_rec:.2%}")
    print(f"  CV F1:            {cv_test_f1:.2%}")
    print(f"  Overfitting Gap:  {gap:.2%} ({'HIGH' if gap > 0.15 else 'OK'})")

best_cv_name = max(cv_results.items(), key=lambda x:x[1]['cv_precision_mean'])[0]
best_cv_metrics = cv_results[best_cv_name]

print("\n" + "="*60)
print(f"BEST MODEL (by CV Precision): {best_cv_name}")
print(f"  CV Precision: {best_cv_metrics['cv_precision_mean']:.2%} (±{best_cv_metrics['cv_precision_std']:.2%}) ")
print(f"  CV Accuracy:  {best_cv_metrics['cv_accuracy']:.2%}")
print(f"  Overfitting:  {best_cv_metrics['train_test_gap']:.2%} gap")
print("="*60 + "\n")

print(f"Retraining best model ({best_cv_name}) on full dataset")

final_vectorizer = TfidfVectorizer(
    max_features=2500,
    ngram_range=(1,2),
    stop_words='english'
)

X_full_vec = final_vectorizer.fit_transform(X_full)
  
if best_cv_name == "LogisticRegression":
    final_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    final_model.fit(X_full_vec, y_full)
elif best_cv_name == "RandomForest":
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(X_full_vec, y_full)
elif best_cv_name == "XGBoost":
    final_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    y_full_encoded = label_enc.fit_transform(y_full)
    final_model.fit(X_full_vec, y_full_encoded)

print(f"Trained {best_cv_name} on full dataset. Also fitted Vectorizer on full dataset")

with mlflow.start_run(run_name=f"{best_cv_name}_CrossValidation"):

    mlflow.log_param("model_type", best_cv_name)
    mlflow.log_param("training_strategy","5-fold_CV")
    mlflow.log_param("trained_on", "full_dataset")
  
    mlflow.log_metric("cv_precision_mean", best_cv_metrics['cv_precision_mean'])
    mlflow.log_metric("cv_precision_std",  best_cv_metrics['cv_precision_std'])
    mlflow.log_metric("cv_accuracy",  best_cv_metrics['cv_accuracy'])
    mlflow.log_metric("cv_recall",  best_cv_metrics['cv_recall'])
    mlflow.log_metric("cv_f1",  best_cv_metrics['cv_f1'])
    mlflow.log_metric("overfitting_gap",  best_cv_metrics['train_test_gap'])

    #logging model
    signature = infer_signature(X_full_vec, final_model.predict(X_full_vec))
    mlflow.sklearn.log_model(final_model, name="model", signature=signature)

    #log vectorizer as artifact
    mlflow.log_artifact("./models/tfidf_vectorizer.pkl")
    mlflow.log_artifact("./models/classifier_metadata.json")

    mlflow.set_tag("note", "Final Production Model - metrics from CV")

#saving model
os.makedirs("./models", exist_ok=True)
joblib.dump(final_model, "./models/document_classifier.pkl")
joblib.dump(final_vectorizer, "./models/tfidf_vectorizer.pkl")

print(f"Registering {final_model} in MLflow model registry...")

client = mlflow.tracking.MlflowClient()
runs = client.search_runs(
    experiment_ids=[client.get_experiment_by_name("document_classifier").experiment_id],
    filter_string = f"tag.mlflow.runName = '{best_cv_name}_CrossValidation'",
    max_results=1
)

if runs:
    run_id = runs[0].info.run_id
    model_uri = f"run:/{run_id}/model"

    #register model
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name="document-classifier"
    )

    print(f"Model registered as 'document-classifier' version {model_version.version}")

    client.transition_model_version_stage(
        name="document-classifier",
        version=model_version.version,
        stage="Production"
    )
    print("Model promoted to production stage")
else:
    print("Could not find run to register")

metadata={
    "model_type": best_cv_name,
    "selection_criterion": "precision",
    "single_split":{
        "precision": float(best_prec),
        "accuracy": float(best_acc),
        "recall": float(best_rec),
        "f1_score": float(best_f1),
    },
    "cross_validation": {
        "cv_precision_mean": float(best_cv_metrics['cv_precision_mean']),
        "cv_precision_std": float(best_cv_metrics['cv_precision_std']),
        "cv_accuracy": float(best_cv_metrics['cv_accuracy']),
        "overfitting_gap": float(best_cv_metrics['train_test_gap'])
    },
    "num_features": X_train_vec.shape[1],
    "categories": df['label'].unique().tolist(),
    "training_samples": len(df)
}

with open("./models/classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Best model: {best_cv_name}")
print("Models saved in './models' directory locally and logged to MLflow!")
print("Run 'mlflow ui' to view experiments")