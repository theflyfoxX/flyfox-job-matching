import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load features
df = pd.read_csv("data/features.csv")

# Drop rows with missing similarity
df = df.dropna(subset=["embedding_similarity"])

# Define features and target
X = df[["embedding_similarity", "location_match", "exp_years_total", "exp_recency_days"]]
y = df["match"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "logreg_model.pkl")
