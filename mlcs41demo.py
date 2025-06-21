import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import pickle
import warnings
import joblib
warnings.filterwarnings("ignore")


# joblibimport 

# Save model, encoders, selected features
# joblib.dump(model_attack, "model_attack.pkl")
# joblib.dump(selected_features, "selected_features.pkl")
# joblib.dump(label_encoders, "label_encoders.pkl")

# # import joblib
# selected_features= {}
# for column in X.select_dtypes(include='object').columns: # type: ignore
#     le = selected_features()
#     Y[column] = le.fit_transform(X[column])
#     selected_features[column] = le

# joblib.dump(selected_features, "selected_features.pkl")

# label_encoders = {}
# for column in X.select_dtypes(include='object').columns: # type: ignore
#     le = LabelEncoder()
#     X[column] = le.fit_transform(X[column])
#     label_encoders[column] = le
    
# # joblib.dump(model_attack, "model_attack.pkl")


# joblib.dump(label_encoders, "label_encoders.pkl")  # if not already saved



# model_attack = {}
# for column in X.select_dtypes(include='object').columns: # type: ignore
#     le = model_attack()
#     X[column] = le.fit_transform(X[column])
#     model_attack[column] = le
# joblib.dump(model_attack, "model_attack.pkl")

# === Load the dataset ===
df = pd.read_csv('cybernew (1).csv')
df.columns = df.columns.str.strip()  # Clean column names



# === Separate targets and features ===
target_attack = 'Attack Type'
target_severity = 'Severity Level'

X = df.drop([target_attack, target_severity], axis=1)
y_attack = df[target_attack]
y_severity = df[target_severity]

# df_input = df_input[selected_features]  # ✅ correct way to apply feature selection
# for col, le in label_encoders.items():
#     df_input[col] = le.transform(df_input[col])


# # import joblib
# selected_features= {}
# for column in X.select_dtypes(include='object').columns: # type: ignore
#     le = selected_features()
#     X[column] = le.fit_transform(X[column])
#     selected_features[column] = le

# joblib.dump(selected_features, "selected_features.pkl")

# label_encoders = {}
# for column in X.select_dtypes(include='object').columns: # type: ignore
#     le = LabelEncoder()
#     X[column] = le.fit_transform(X[column])
#     label_encoders[column] = le
    
# # joblib.dump(model_attack, "model_attack.pkl")


# joblib.dump(label_encoders, "label_encoders.pkl")  # if not already saved



# model_attack = {}
# for column in X.select_dtypes(include='object').columns: # type: ignore
#     le = model_attack()
#     X[column] = le.fit_transform(X[column])
#     model_attack[column] = le
# joblib.dump(model_attack, "model_attack.pkl")

# df_input = df_input[selected_features]  # ✅ correct way to apply feature selection
# for col, le in label_encoders.items():
#     df_input[col] = le.transform(df_input[col])
# # === Encode categorical features ===
label_encoders = {}
for column in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# === Encode target columns ===
y_attack_encoder = LabelEncoder()
y_attack = y_attack_encoder.fit_transform(y_attack)
label_encoders['Attack Type'] = y_attack_encoder

y_severity_encoder = LabelEncoder()
y_severity = y_severity_encoder.fit_transform(y_severity)
label_encoders['Severity Level'] = y_severity_encoder

# === Feature selection ===
sfm = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
sfm.fit(X, y_attack)
selected_features = X.columns[sfm.get_support()]
X = X[selected_features]

# === Train-test split ===
X_train, X_test, y_attack_train, y_attack_test, y_severity_train, y_severity_test = train_test_split(
    X, y_attack, y_severity, test_size=0.2, random_state=42
)

# === Hyperparameter tuning for Attack model ===
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

rfc = RandomForestClassifier(random_state=42)
search_attack = RandomizedSearchCV(rfc, param_grid, n_iter=10, cv=3)
search_attack.fit(X_train, y_attack_train)
model_attack = search_attack.best_estimator_

# === Train Severity model ===
model_severity = RandomForestClassifier(n_estimators=100, random_state=42)
model_severity.fit(X_train, y_severity_train)

# === Evaluate models ===
print("\n--- Attack Type Model ---")
y_attack_pred = model_attack.predict(X_test)
print("Accuracy:", accuracy_score(y_attack_test, y_attack_pred))
print("Classification Report:\n", classification_report(y_attack_test, y_attack_pred))

print("\n--- Severity Level Model ---")
y_severity_pred = model_severity.predict(X_test)
print("Accuracy:", accuracy_score(y_severity_test, y_severity_pred))
print("Classification Report:\n", classification_report(y_severity_test, y_severity_pred))

# === Confusion Matrix ===
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_attack_test, y_attack_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix for Attack Type")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# === Feature Importance ===
importances = model_attack.feature_importances_
sorted_idx = importances.argsort()
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_idx], y=X.columns[sorted_idx])
plt.title("Feature Importance for Attack Type")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# === Take user input for prediction ===
print("\nEnter the following details for prediction:")
user_input = {}

original_columns = df.drop(columns=[target_attack, target_severity]).columns
for col in original_columns:
    le = label_encoders.get(col)
    if le:  # Categorical
        print(f"Options for {col}: {list(le.classes_)}")
        val = input(f"{col}: ")
        try:
            user_input[col] = le.transform([val])[0]
        except ValueError:
            print(f"Invalid input for {col}. Available options: {list(le.classes_)}")
            exit()
    else:  # Numeric
        val = input(f"{col} (numeric,e.g.,1-100 if normalized): ")  #, e.g., 0.0 - 1.0 if normalized
        try:
            user_input[col] = float(val)
        except ValueError:
            print(f"Invalid numeric input for {col}.")
            exit()

# === Convert input to DataFrame ===
user_input_df = pd.DataFrame([user_input])[original_columns]
user_input_df = user_input_df[selected_features]

# === Make predictions ===
pred_attack = model_attack.predict(user_input_df)
pred_severity = model_severity.predict(user_input_df)

attack_label = y_attack_encoder.inverse_transform(pred_attack)[0]
severity_label = y_severity_encoder.inverse_transform(pred_severity)[0]

print("\n--- Prediction Result ---")
print("Predicted Attack Type:", attack_label)
print("Predicted Severity Level:", severity_label)

# === Save prediction to log ===
result = {**user_input, 'Predicted Attack Type': attack_label, 'Predicted Severity Level': severity_label}
pd.DataFrame([result]).to_csv('prediction_log.csv', mode='a', index=False, header=False)

# === Save both models ===
with open('attack_type_model.pkl', 'wb') as f:
    pickle.dump(model_attack, f)
with open('severity_level_model.pkl', 'wb') as f:
    pickle.dump(model_severity, f)

    
#Logging every prediction in a csv file:-
result = {
    **user_input,
    'Predicted Attack Type': attack_label,
    'Predicted Severity Level': severity_label
}
pd.DataFrame([result]).to_csv('prediction_log.csv', mode='a', index=False, header=False)
    

print("\nModels saved as 'attack_type_model.pkl' and 'severity_level_model.pkl' ,'label_encoders.pkl', 'selected_features.pkl'")