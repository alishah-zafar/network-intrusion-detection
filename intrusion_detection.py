# ============================================================
# Network Intrusion Detection using NSL-KDD dataset
# Goal: look at network traffic and predict "normal" or "attack"
# ============================================================

# ---------- 1. IMPORTS ----------
import pandas as pd                                   # loads and handles table data
import numpy as np                                    # [NEW] used for the feature importance plot
import joblib                                         # [NEW] saves the trained model to a file
import matplotlib                                    # [NEW] used for making plots
matplotlib.use("Agg")                                 # [NEW] lets plots save to files without a window
import matplotlib.pyplot as plt                       # [NEW] plotting

from sklearn.compose import ColumnTransformer         # applies different steps to different columns
from sklearn.preprocessing import OneHotEncoder       # converts words (categories) into numbers
from sklearn.ensemble import RandomForestClassifier   # the model
from sklearn.pipeline import Pipeline                 # chains preprocessing + model together
from sklearn.metrics import (
    accuracy_score,                                   # % of correct predictions
    classification_report,                            # precision, recall, F1-score
    confusion_matrix,                                 # correct vs wrong predictions per class
    recall_score,                                     # how many real attacks we caught
    ConfusionMatrixDisplay,                           # [NEW] draws the confusion matrix as an image
)

# ---------- 2. COLUMN NAMES ----------
# The dataset file has no header row, so we give names to each column
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

# ---------- 3. HELPER FUNCTION ----------
def simplify_label(value):
    """The dataset has many attack names (neptune, smurf...).
    We turn them all into just 'attack'. 'normal' stays 'normal'."""
    if value == "normal":
        return "normal"
    else:
        return "attack"

# ---------- 4. LOAD TRAINING DATA ----------
data = pd.read_csv("KDDTrain+.txt", header=None)      # header=None -> file has no column names
data.columns = columns                                # give names to each column

# .apply() runs simplify_label on every row of the 'label' column
data["binary_label"] = data["label"].apply(simplify_label)
print("Training data class counts:")
print(data["binary_label"].value_counts())

# y = the correct answers (what we want to predict)
y = data["binary_label"]
# X = the inputs (everything except the answer columns)
X = data.drop(columns=["label", "binary_label", "difficulty"])

# ---------- 5. LOAD TEST DATA ----------
# Same steps as training data. This data is never shown to the model during training.
test_data = pd.read_csv("KDDTest+.txt", header=None)
test_data.columns = columns
test_data["binary_label"] = test_data["label"].apply(simplify_label)

X_test = test_data.drop(columns=["label", "binary_label", "difficulty"])
y_test = test_data["binary_label"]

# ---------- 6. BUILD THE MODEL ----------
# These 3 columns contain words, so they must be converted to numbers
categorical_columns = ["protocol_type", "service", "flag"]

preprocessor = ColumnTransformer(
    transformers=[
        # handle_unknown='ignore' -> if the test data has a category not seen
        # in training, don't crash, just ignore it
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ],
    remainder="passthrough"   # all other columns are already numbers, keep them as they are
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,         # use 100 decision trees
        random_state=42,          # makes results the same every time you run it
        n_jobs=-1,                # use all CPU cores (faster)
        class_weight="balanced"   # give fair importance to both classes
    ))
])

# ---------- 7. TRAIN ----------
model.fit(X, y)   # model learns from traffic features (X) and correct labels (y)
print("\nModel training completed")

# ---------- 8. PREDICT ON TEST DATA ----------
y_pred = model.predict(X_test)   # model's guesses on data it has never seen

# ---------- 9. EVALUATE ----------
accuracy = accuracy_score(y_test, y_pred)
attack_recall = recall_score(y_test, y_pred, pos_label="attack")

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("Attack Detection Rate:", round(attack_recall * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix layout:
#                  predicted normal | predicted attack
# actual normal   [      cm[0][0]   |      cm[0][1]    ]
# actual attack   [      cm[1][0]   |      cm[1][1]    ]
cm = confusion_matrix(y_test, y_pred, labels=["normal", "attack"])
print("Confusion Matrix:")
print(cm)

print("\nDetailed Results:")
print("Normal records correctly identified:", cm[0][0])
print("Normal records incorrectly flagged as attacks:", cm[0][1])
print("Attacks incorrectly classified as normal:", cm[1][0])
print("Attacks correctly detected:", cm[1][1])

# [NEW] False negative rate = % of real attacks we MISSED.
# In security this is the most dangerous mistake, so it's worth reporting.
missed = cm[1][0]
total_attacks = cm[1][0] + cm[1][1]
print("Missed attack rate:", round(missed / total_attacks * 100, 2), "%")

# ---------- 10. SAVE PLOTS ----------
# [NEW] Confusion matrix as an image (good for your README)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=["normal", "attack"])
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# [NEW] Which features did the model rely on most?
rf_model = model.named_steps["classifier"]                          # get the Random Forest out of the pipeline
feature_names = model.named_steps["preprocessor"].get_feature_names_out()  # names after one-hot encoding
importances = rf_model.feature_importances_
top_15 = np.argsort(importances)[-15:]                              # positions of the 15 biggest values

plt.figure(figsize=(8, 6))
plt.barh(feature_names[top_15], importances[top_15])
plt.title("Top 15 Important Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# ---------- 11. SAVE THE MODEL ----------
# [NEW] So you can reuse the trained model later without retraining
joblib.dump(model, "intrusion_model.pkl")

print("\nPrediction completed successfully!")
print("Total test records:", len(X_test))
print("Total predictions:", len(y_pred))
print("Saved: confusion_matrix.png, feature_importance.png, intrusion_model.pkl")