import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# === Load the dataset ===
df = pd.read_csv('cybernew (1).csv')

# Strip column names to remove leading/trailing spaces
df.columns = df.columns.str.strip()

# === Set target and features ===
y = df['Attack Type']                      # Target variable
X = df.drop('Attack Type', axis=1)        # Features (excluding target)

# === Encode categorical features ===
label_encoders = {}
for column in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# === Encode the target (Attack Type) ===
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)
label_encoders['Attack Type'] = y_encoder

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Train the Random Forest model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate the model ===
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === Take user input for prediction ===
print("\nEnter the following details for attack type prediction:")
user_input = {}

# Input columns (excluding the target)
input_columns = ['Protocol', 'Packet Type', 'Traffic Type', 'Severity Level']

for col in input_columns:
    val = input(f"{col}: ")
    le = label_encoders.get(col)
    if le:
        try:
            user_input[col] = le.transform([val])[0]
        except ValueError:
            print(f"Invalid input for {col}. Available options: {list(le.classes_)}")
            exit()
    else:
        print(f"No encoder found for column: {col}")
        exit()

# Input numeric feature: Anomaly Scores
try:
    user_input['Anomaly Scores'] = float(input("Anomaly Scores (e.g., 0.15): "))
except ValueError:
    print("Invalid input for Anomaly Scores. Please enter a numeric value.")
    exit()

# === Ensure correct column order and predict ===
input_df = pd.DataFrame([user_input])[X.columns]  # Match training columns
predicted = model.predict(input_df)
predicted_label = y_encoder.inverse_transform(predicted)
print("\nPredicted Attack Type:", predicted_label[0])

# === Save model to pickle file ===
with open('attack_type_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as 'attack_type_model.pkl'")











































































































































































# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score

# # Load the dataset
# df = pd.read_csv('cybernew.csv')

# # Separate features and target
# X = df.drop('Severity Level', axis=1)
# y = df['Severity Level']

# # Encode categorical features
# label_encoders = {}
# for column in X.select_dtypes(include='object').columns:
#     le = LabelEncoder()
#     X[column] = le.fit_transform(X[column])
#     label_encoders[column] = le

# # Encode target    TCP,data,DNS,Intrusion,78.5,High
# y_encoder = LabelEncoder()
# y = y_encoder.fit_transform(y)
# label_encoders['Severity Level'] = y_encoder

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Train Random Forest model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Evaluate model
# y_pred = model.predict(X_test)
# print("\nAccuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # === Take User Input for Prediction ===
# print("\nEnter the following details for prediction:")

# user_input = {}
# for col in ['Protocol', 'Packet Type', 'Traffic Type', 'Attack Type']:
#     val = input(f"{col}: ")
#     le = label_encoders[col]
#     try:
#         user_input[col] = le.transform([val])[0]
#     except ValueError:
#         print(f"Invalid input for {col}. Available options: {list(le.classes_)}")
#         exit()

# # Get anomaly score
# try:
#     user_input['Anomaly Scores'] = float(input("Anomaly Scores (e.g., 0.15): "))
# except ValueError:
#     print("Invalid input for Anomaly Scores. Please enter a numeric value.")
#     exit()

# # Convert input to DataFrame
# input_df = pd.DataFrame([user_input])

# # Predict
# predicted = model.predict(input_df)
# predicted_label = y_encoder.inverse_transform(predicted)
# print("\nPredicted Severity Level:", predicted_label[0])
# import pickle
# with open('model_pickle,','wb') as f:
#     pickle.dump(model,f)