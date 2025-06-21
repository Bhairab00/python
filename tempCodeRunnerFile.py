#Logging every prediction in a csv file:-
result = {
    **user_input,
    'Predicted Attack Type': attack_label,
    'Predicted Severity Level': severity_label
}
pd.DataFrame([result]).to_csv('prediction_log.csv', mode='a', index=False, header=False)