import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("student_old_one.csv")

# Show column names (for debug)
print("Columns in your CSV:", df.columns)

# Convert gender to numeric
df['gender'] = df['gender'].map({'female': 0, 'male': 1})

# ✅ Replace 'score' below with your actual output column name if needed
X = df[['study_time', 'attendance', 'gender']]
y = df['score']  # <-- change this if your target column is different

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.pkl')
print("✅ Model trained and saved as model.pkl")
