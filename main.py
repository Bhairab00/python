from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import pickle
import joblib
import os

app = FastAPI()

# Ensure static directory exists
if not os.path.exists("static"):
    os.mkdir("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load models and preprocessing artifacts
with open("attack_type_model.pkl", "rb") as f:
    model_attack = pickle.load(f)

with open("severity_level_model.pkl", "rb") as f:
    model_severity = pickle.load(f)

selected_features = joblib.load("selected_features.pkl")
label_encoders = joblib.load("label_encoders.pkl")

@app.get("/", response_class=HTMLResponse)
def form_post(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    Protocol: str = Form(...),
    PacketType: str = Form(...),
    TrafficType: str = Form(...),
    AnomalyScore: float = Form(...)
):
    # Collect user input into a dictionary
    input_dict = {
        "Protocol": Protocol,
        "PacketType": PacketType,
        "TrafficType": TrafficType,
        "Anomaly Scores": AnomalyScore
    }

    # Convert input to DataFrame
    df_input = pd.DataFrame([input_dict])

    # Apply encoders only to matching columns
    for col in df_input.columns:
        if col in label_encoders:
            df_input[col] = label_encoders[col].transform(df_input[col])

    # Select only relevant features
    df_input_selected = df_input[selected_features]

    # Make predictions
    pred_attack = model_attack.predict(df_input_selected)[0]
    pred_severity = model_severity.predict(df_input_selected)[0]
    
    print("Expected features by model:", selected_features)
    print("User input columns:", df_input.columns)


    # Inverse transform predictions to labels
    attack_label = label_encoders["Attack Type"].inverse_transform([pred_attack])[0]
    severity_label = label_encoders["Severity Level"].inverse_transform([pred_severity])[0]


    #    # Return result
    # return templates.TemplateResponse("result.html", {
    # "request": request,
    # "prediction": f"{attack_label} (Severity: {severity_label})"
    # })
    
    
    
    
    # ✅ Log the result to CSV
    result = {
        **input_dict,
        'Predicted Attack Type': attack_label,
        'Predicted Severity Level': severity_label
    }

    # Append to CSV file
    pd.DataFrame([result]).to_csv('prediction_log.csv', mode='a', index=False, header=not os.path.exists('prediction_log.csv'))
    
    # Return result to the user
    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": f"{attack_label} (Severity: {severity_label})"
        # "pd.DataFrame([result]).to_csv('prediction_log.csv')"
    })
    