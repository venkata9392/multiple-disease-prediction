import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("D:\\multiple\\heart.csv")

# Features and Target
X = data.drop("target", axis=1)
Y = data["target"]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Accuracy
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Heart Training Accuracy:", accuracy_score(Y_train, train_pred))
print("Heart Testing Accuracy:", accuracy_score(Y_test, test_pred))

# Save model
pickle.dump(model, open("heart_model.sav", "wb"))

print("Heart model saved successfully")
