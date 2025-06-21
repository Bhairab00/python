import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


from xgboost import XGBClassifier
model_attack = XGBClassifier()
model_severity = XGBClassifier()


# # Pie chart for predicted outputs
# labels = ['Attack Type', 'Severity Level']
# values = [1, 1]  # Static 1 for simple visual

# plt.figure(figsize=(6, 6))
# plt.pie(values, labels=[attack_label, severity_label], autopct='%1.1f%%', colors=['orange', 'red'])
# plt.title("Predicted Attack and Severity Level")
# plt.savefig("prediction_pie_chart.png")
# plt.show()

# === Load the dataset ===
df = pd.read_csv('cybernew (1).csv')
df.columns = df.columns.str.strip()  # Clean column names

# === Separate targets and features ===
target_attack = 'Attack Type'
target_severity = 'Severity Level'

X = df.drop([target_attack, target_severity], axis=1)
y_attack = df[target_attack]
y_severity = df[target_severity]

# === Encode categorical features ===
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

# === Train-test split ===
X_train, X_test, y_attack_train, y_attack_test, y_severity_train, y_severity_test = train_test_split(
    X, y_attack, y_severity, test_size=0.2, random_state=42
)

# === Train Random Forest models ===
model_attack = RandomForestClassifier(n_estimators=100, random_state=42)
model_attack.fit(X_train, y_attack_train)

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

# === Take user input for prediction ===
print("\nEnter the following details for prediction:")
user_input = {}

input_columns = X.columns.tolist()

for col in input_columns:
    le = label_encoders.get(col)
    if le:  # Categorical
        val = input(f"{col}: ")
        try:
            user_input[col] = le.transform([val])[0]
        except ValueError:
            print(f"Invalid input for {col}. Available options: {list(le.classes_)}")
            exit()
    else:  # Numerical
        try:
            user_input[col] = float(input(f"{col} (numeric): "))
        except ValueError:
            print(f"Invalid numeric input for {col}.")
            exit()

# === Convert input to DataFrame ===
input_df = pd.DataFrame([user_input])[X.columns]  # Ensure same order

# === Make predictions ===
pred_attack = model_attack.predict(input_df)
pred_severity = model_severity.predict(input_df)

attack_label = y_attack_encoder.inverse_transform(pred_attack)[0]
severity_label = y_severity_encoder.inverse_transform(pred_severity)[0]

print("\n--- Prediction Result ---")
print("Predicted Attack Type:", attack_label)
print("Predicted Severity Level:", severity_label)

# === Save both models ===
with open('attack_type_model.pkl', 'wb') as f:
    pickle.dump(model_attack, f)
with open('severity_level_model.pkl', 'wb') as f:
    pickle.dump(model_severity, f)

print("\nModels saved as 'attack_type_model.pkl' and 'severity_level_model.pkl'")


labels = ['Attack Type', 'Severity Level']
values = [1, 1]  # Static 1 for simple visual

plt.figure(figsize=(6, 6))
plt.pie(values, labels=[attack_label, severity_label], autopct='%1.1f%%', colors=['orange', 'red'])
plt.title("Predicted Attack and Severity Level")
plt.savefig("prediction_pie_chart.png")
plt.show()


# ######barchar
# attack_probs = model_attack.predict_proba(input_df)[0]
# severity_probs = model_severity.predict_proba(input_df)[0]

# attack_classes = y_attack_encoder.inverse_transform(range(len(attack_probs)))
# severity_classes = y_severity_encoder.inverse_transform(range(len(severity_probs)))

# print("\nAttack Type Probabilities:")
# for cls, prob in zip(attack_classes, attack_probs):
#     print(f"{cls}: {prob:.2f}")

# print("\nSeverity Level Probabilities:")
# for cls, prob in zip(severity_classes, severity_probs):
#     print(f"{cls}: {prob:.2f}")


##ANAMOLY
# from sklearn.ensemble import IsolationForest

# Train it during preprocessing
anomaly_detector = IsolationForest(contamination=0.05)
anomaly_detector.fit(X)

# Use before prediction
if anomaly_detector.predict(input_df)[0] == -1:
    print("Warning: Input may be anomalous or unseen.")


