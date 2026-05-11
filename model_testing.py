import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CREATE OUTPUT DIRECTORIES
# ==========================================
os.makedirs("plots/confusion_matrices", exist_ok=True)
os.makedirs("plots/model_comparisons", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ==========================================
# 2. LOAD CSV FILE
# ==========================================
CSV_PATH = "asl_landmark_features.csv"

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)

print(df.head())
print("\nDataset Shape:", df.shape)

# ==========================================
# 3. SEPARATE FEATURES AND LABELS
# ==========================================
X = df.drop("label", axis=1)
y = df["label"]

# ==========================================
# 4. ENCODE LABELS
# ==========================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nClasses:")
print(label_encoder.classes_)

# ==========================================
# 5. TRAIN-TEST SPLIT (80:20 PER LABEL)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

print("\nTraining Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

# ==========================================
# 6. FEATURE SCALING
# ==========================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# 7. DEFINE MODELS
# ==========================================
models = {
    "SVM": SVC(kernel='rbf'),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# ==========================================
# 8. TRAIN AND EVALUATE MODELS
# ==========================================
results = []

for model_name, model in models.items():

    print("\n" + "="*50)
    print(f"TRAINING: {model_name}")
    print("="*50)

    # ==========================================
    # TRAIN MODEL
    # ==========================================
    model.fit(X_train, y_train)

    # ==========================================
    # PREDICT
    # ==========================================
    y_pred = model.predict(X_test)

    # ==========================================
    # METRICS
    # ==========================================
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average='weighted'
    )

    recall = recall_score(
        y_test,
        y_pred,
        average='weighted'
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted'
    )

    # ==========================================
    # STORE RESULTS
    # ==========================================
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    # ==========================================
    # PRINT RESULTS
    # ==========================================
    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    # ==========================================
    # CLASSIFICATION REPORT
    # ==========================================
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )

    print("\nClassification Report:")
    print(report)

    # Save report to text file
    report_path = f"reports/{model_name}_classification_report.txt"

    with open(report_path, "w") as f:
        f.write(report)

    # ==========================================
    # CONFUSION MATRIX
    # ==========================================
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )

    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()

    # Save confusion matrix
    cm_path = f"plots/confusion_matrices/{model_name}_confusion_matrix.png"

    plt.savefig(cm_path, dpi=300, bbox_inches='tight')

    plt.show()

    plt.close()

# ==========================================
# 9. RESULTS COMPARISON TABLE
# ==========================================
results_df = pd.DataFrame(results)

print("\n" + "="*50)
print("FINAL MODEL COMPARISON")
print("="*50)

print(results_df)

# ==========================================
# 10. SAVE RESULTS CSV
# ==========================================
results_df.to_csv(
    "reports/model_results.csv",
    index=False
)

# ==========================================
# 11. VISUALIZE METRICS
# ==========================================
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

for metric in metrics:

    plt.figure(figsize=(8, 5))

    sns.barplot(
        x="Model",
        y=metric,
        data=results_df
    )

    plt.title(f"Model Comparison - {metric}")
    plt.ylim(0, 1)

    plt.tight_layout()

    # Save plot
    plot_path = f"plots/model_comparisons/{metric}_comparison.png"

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    plt.show()

    plt.close()

# ==========================================
# 12. OVERALL COMPARISON GRAPH
# ==========================================
results_df.set_index("Model")[metrics].plot(
    kind='bar',
    figsize=(10, 6)
)

plt.title("Overall Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)

plt.tight_layout()

# Save overall comparison graph
overall_plot_path = "plots/model_comparisons/overall_model_comparison.png"

plt.savefig(overall_plot_path, dpi=300, bbox_inches='tight')

plt.show()

plt.close()

print("\n===================================")
print("ALL RESULTS SAVED SUCCESSFULLY")
print("===================================")

print("\nSaved Files:")

print("\nReports:")
print("- reports/model_results.csv")
print("- reports/*_classification_report.txt")

print("\nConfusion Matrices:")
print("- plots/confusion_matrices/")

print("\nComparison Graphs:")
print("- plots/model_comparisons/")