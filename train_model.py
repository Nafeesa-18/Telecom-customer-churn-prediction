import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("churn.csv")

# Remove unnecessary columns
drop_cols = [
    "Customer ID",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Score",
    "Churn Category",
    "Churn Reason"
]

df.drop(columns=drop_cols, inplace=True)

# Handle missing values
df["Offer"] = df["Offer"].fillna("None")
df["Internet Type"] = df["Internet Type"].fillna("No Internet")

# Create target column
df["Customer Status"] = df["Customer Status"].map({
    "Stayed": 0,
    "Joined": 0,
    "Churned": 1
})

# Encode all non-numeric columns
encoders = {}

for col in df.columns:
    if df[col].dtype != "int64" and df[col].dtype != "float64":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        joblib.dump(encoders,"encoders.pkl")
print(df.columns.tolist())
# Features and Target
X = df[
    [
        "Gender",
        "Age",
        "Tenure in Months",
        "Offer",
        "Internet Type",
        "Contract",
        "Payment Method",
        "Monthly Charge",
        "Satisfaction Score",
        "CLTV"
    ]
]
y = df["Customer Status"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

#save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")
print("Model saved successfully")

