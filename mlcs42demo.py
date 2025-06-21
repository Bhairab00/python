# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from sklearn.model_selection import train_test_split, RandomizedSearchCV
# # from sklearn.preprocessing import LabelEncoder
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# # from sklearn.feature_selection import SelectFromModel
# # import pickle
# # import joblib
# # import warnings
# # warnings.filterwarnings("ignore")
# # df = pd.read_csv("cybernew (1).csv")
# # df.columns = df.columns.str.strip() # Remove whitespace from column names
# # target_attack = "Attack Type"
# # target_severity = "Severity Level"
# # X = df.drop([target_attack, target_severity], axis=1)
# # y_attack = df[target_attack]
# # y_severity = df[target_severity]
# # label_encoders = {}
# # for column in X.select_dtypes(include="object").columns:
# #   le = LabelEncoder()
# # X[column] = le.fit_transform(X[column])
# # label_encoders[column] = le
# # y_attack_encoder = LabelEncoder()
# # y_attack = y_attack_encoder.fit_transform(y_attack)
# # label_encoders["Attack Type"] = y_attack_encoder
# # y_severity_encoder = LabelEncoder()
# # y_severity = y_severity_encoder.fit_transform(y_severity)
# # label_encoders["Severity Level"] = y_severity_encoder
# # sfm = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
# # sfm.fit(X, y_attack)
# # selected_features = X.columns[sfm.get_support()]
# # X_selected = X[selected_features]
# # X_train, X_test, y_attack_train, y_attack_test, y_severity_train, y_severity_test = train_test_split(
# # X_selected, y_attack, y_severity, test_size=0.2, random_state=42
# # )
# # param_grid = {
# # "n_estimators": [100, 200],
# # "max_depth": [10, 20, None],
# # "min_samples_split": [2, 5],
# # "min_samples_leaf": [1, 2],
# # "bootstrap": [True, False],
# # }
# # rfc = RandomForestClassifier(random_state=42)
# # search_attack = RandomizedSearchCV(rfc, param_grid, n_iter=10, cv=3)
# # search_attack.fit(X_train, y_attack_train)
# # model_attack = search_attack.best_estimator_

# # model_severity = RandomForestClassifier(n_estimators=100, random_state=42)
# # model_severity.fit(X_train, y_severity_train)

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# from sklearn.feature_selection import SelectFromModel
# import joblib
# import pickle
# import warnings

# warnings.filterwarnings("ignore")
# df = pd.read_csv("cybernew (1).csv")
# df.columns = df.columns.str.strip()
# target_attack = "Attack Type"
# target_severity = "Severity Level"
# X = df.drop([target_attack, target_severity], axis=1)
# y_attack = df[target_attack]
# y_severity = df[target_severity]
# label_encoders = {}
# for col in X.select_dtypes(include="object").columns:
#  le = LabelEncoder()
# X[col] = le.fit_transform(X[col])
# label_encoders[col] = le
# le_attack = LabelEncoder()
# y_attack = le_attack.fit_transform(y_attack)
# label_encoders["Attack Type"] = le_attack

# le_severity = LabelEncoder()
# y_severity = le_severity.fit_transform(y_severity)
# label_encoders["Severity Level"] = le_severity
# sfm = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
# # sfm.fit(X, y_attack)
# selected_features = X.columns[sfm.get_support()]
# X_selected = X[selected_features]


# label_encoders = {}
# for column in X.select_dtypes(include="object").columns:
#   le = LabelEncoder()
# X[column] = le.fit_transform(X[column])
# label_encoders[column] = le
# y_attack_encoder = LabelEncoder()
# y_attack = y_attack_encoder.fit_transform(y_attack)
# label_encoders["Attack Type"] = y_attack_encoder
# y_severity_encoder = LabelEncoder()
# y_severity = y_severity_encoder.fit_transform(y_severity)
# label_encoders["Severity Level"] = y_severity_encoder
# sfm = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
# sfm.fit(X, y_attack)
# selected_features = X.columns[sfm.get_support()]
# X_selected = X[selected_features]
# X_train, X_test, y_attack_train, y_attack_test, y_severity_train, y_severity_test = train_test_split(
# X_selected, y_attack, y_severity, test_size=0.2, random_state=42
# )

# param_grid = {
# "n_estimators": [100, 200],
# "max_depth": [10, 20, None],
# "min_samples_split": [2, 5],
# "min_samples_leaf": [1, 2],
# "bootstrap": [True, False],
# }
# rfc = RandomForestClassifier(random_state=42)
# search_attack = RandomizedSearchCV(rfc, param_grid, n_iter=10, cv=3)
# search_attack.fit(X_train, y_attack_train)
# model_attack = search_attack.best_estimator_

# model_severity = RandomForestClassifier(n_estimators=100, random_state=42)
# model_severity.fit(X_train, y_severity_train)






# print("\n--- Attack Type Model Evaluation ---")
# y_attack_pred = model_attack.predict(X_test)
# print("Accuracy:", accuracy_score(y_attack_test, y_attack_pred))
# print("Classification Report:\n", classification_report(y_attack_test, y_attack_pred))

# print("\n--- Severity Level Model Evaluation ---")
# y_severity_pred = model_severity.predict(X_test)
# print("Accuracy:", accuracy_score(y_severity_test, y_severity_pred))
# print("Classification Report:\n", classification_report(y_severity_test, y_severity_pred))

# plt.figure(figsize=(6, 4))
# sns.heatmap(confusion_matrix(y_attack_test, y_attack_pred), annot=True, fmt="d", cmap="Blues")
# plt.title("Confusion Matrix for Attack Type")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# importances = model_attack.feature_importances_
# sorted_idx = importances.argsort()
# plt.figure(figsize=(10, 6))
# sns.barplot(x=importances[sorted_idx], y=X_selected.columns[sorted_idx])
# plt.title("Feature Importance for Attack Type")
# plt.xlabel("Importance Score")
# plt.ylabel("Feature")
# plt.show()

# with open("attack_type_model.pkl", "wb") as f:
#   pickle.dump(model_attack, f)

# with open("severity_level_model.pkl", "wb") as f:
#   pickle.dump(model_severity, f)

# joblib.dump(label_encoders, "label_encoders.pkl")
# joblib.dump(selected_features.tolist(), "selected_features.pkl")

# print("\n✅ Models and encoders saved:")
# print("- attack_type_model.pkl")
# print("- severity_level_model.pkl")
# print("- label_encoders.pkl")
# print("- selected_features.pkl")







import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import joblib
import pickle
import warnings

warnings.filterwarnings("ignore")

# === Load dataset ===
df = pd.read_csv("cybernew (1).csv")
df.columns = df.columns.str.strip()

# === Define targets ===
target_attack = "Attack Type"
target_severity = "Severity Level"

# Separate features and targets
X = df.drop([target_attack, target_severity], axis=1)
y_attack = df[target_attack]
y_severity = df[target_severity]

# === Encode categorical features ===
label_encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode targets
le_attack = LabelEncoder()
y_attack = le_attack.fit_transform(y_attack)
label_encoders["Attack Type"] = le_attack

le_severity = LabelEncoder()
y_severity = le_severity.fit_transform(y_severity)
label_encoders["Severity Level"] = le_severity

# === Feature Selection ===
sfm = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
sfm.fit(X, y_attack)
selected_features = X.columns[sfm.get_support()]
X_selected = X[selected_features]

# === Train-test split ===
X_train, X_test, y_attack_train, y_attack_test, y_severity_train, y_severity_test = train_test_split(
    X_selected, y_attack, y_severity, test_size=0.2, random_state=42
)

# === Hyperparameter tuning for Attack model ===
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "bootstrap": [True, False],
}
rfc = RandomForestClassifier(random_state=42)
search_attack = RandomizedSearchCV(rfc, param_grid, n_iter=10, cv=3)
search_attack.fit(X_train, y_attack_train)
model_attack = search_attack.best_estimator_

# === Train Severity model ===
model_severity = RandomForestClassifier(n_estimators=100, random_state=42)
model_severity.fit(X_train, y_severity_train)

# === Evaluate models ===
print("\\n--- Attack Type Model ---")
y_attack_pred = model_attack.predict(X_test)
print("Accuracy:", accuracy_score(y_attack_test, y_attack_pred))
print("Classification Report:\\n", classification_report(y_attack_test, y_attack_pred))

print("\\n--- Severity Level Model ---")
y_severity_pred = model_severity.predict(X_test)
print("Accuracy:", accuracy_score(y_severity_test, y_severity_pred))
print("Classification Report:\\n", classification_report(y_severity_test, y_severity_pred))

# === Plot Confusion Matrix ===
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_attack_test, y_attack_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix for Attack Type")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# === Feature Importance Plot ===
importances = model_attack.feature_importances_
sorted_idx = importances.argsort()
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_idx], y=X_selected.columns[sorted_idx])
plt.title("Feature Importance for Attack Type")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# === Save models and encoders ===
with open("attack_type_model.pkl", "wb") as f:
    pickle.dump(model_attack, f)

with open("severity_level_model.pkl", "wb") as f:
    pickle.dump(model_severity, f)

joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(selected_features.tolist(), "selected_features.pkl")

print("\\n✅ All models and encoders saved:")
print("- attack_type_model.pkl")
print("- severity_level_model.pkl")
print("- label_encoders.pkl")
print("- selected_features.pkl")

