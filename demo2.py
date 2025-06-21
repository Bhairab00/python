import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import os
for dirname, _, filenames in os.walk('Final Yr Project'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
       
df=pd.read_csv("cybernew.csv")
# print(df)
df.isnull().sum()
# print(df.columns)

print(df.columns)
anomaly_scores = df['Anomaly Scores']
plt.figure(figsize=(8, 6))
sns.boxplot(x=anomaly_scores)
plt.title('Distribution of Anomaly Scores (Box Plot)')
plt.xlabel('Anomaly Scores')
plt.show()
plt.figure(figsize=(8, 6))
sns.histplot(anomaly_scores, kde=True, bins=20, color='green')
plt.title('Distribution of Anomaly Scores (Histogram)')
plt.xlabel('Anomaly Scores')
plt.ylabel('Frequency')
plt.show()
sns.pointplot(data=df, x="Severity Level", y="Anomaly Scores", hue="Anomaly Scores", markers="o", linestyles="")
plt.title("Severity Level vs Anomaly Scores")
plt.xlabel("Severity Level")
plt.ylabel("Anomaly Scores")
plt.legend(title="Anomaly Scores", loc="upper right")
plt.show()
#for Categorical Values
from sklearn.preprocessing import OneHotEncoder
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(df[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
df_encoded = pd.concat([df, one_hot_df], axis=1)
#for Numeric Values
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from collections import Counter
df_encoded['Anomaly Scores'] = (df_encoded['Anomaly Scores'] - df_encoded['Anomaly Scores'].mean()) / df_encoded['Anomaly Scores'].std() 

########### print(df_encoded) for check name of traffic type
#print(df_encoded.head(4))
print(df_encoded.columns)
# df_vbf=df_encoded
# df_vbf.drop(['Attack Type','Attack Type_DDoS','Attack Type_Malware','Attack Type_Intrusion'],axis=1,inplace=True)
# print(df_vbf)
# #print(f"Encoded Employee data : \n{df_encoded}")
df_var = df_encoded
df1=df_var[['Anomaly Scores','Protocol_ICMP', 'Protocol_TCP',
       'Protocol_UDP', 'Packet Type_control', 'Packet Type_data',
       'Traffic Type_DNS', 'Traffic Type_FTP', 'Traffic Type_HTTP',
       'Attack Type_DDoS', 'Attack Type_Intrusion', 'Attack Type_Malware']]
df2=df_var[['Severity Level']]

df1_train, df1_test, df2_train, df2_test = train_test_split(df1, df2, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(df1_train, df2_train)

y_pred = clf.predict(df1_test)
print(f"Accuracy: {accuracy_score(df2_test, y_pred)}")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Example true labels and predicted labels

# Generate the confusion matrix
cm = confusion_matrix(df2_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Reds)

# Add titles and labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted Severity Label')
plt.ylabel('True Severity Label')
plt.show()
