import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("D:\\multiple\\parkinsons.csv")

# Drop name column
data = data.drop("name", axis=1)

# Features and Target
X = data.drop("status", axis=1)
Y = data["status"]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2)

# Train model
model = SVC(kernel='linear')
model.fit(X_train, Y_train)

# Accuracy
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Parkinson's Training Accuracy:", accuracy_score(Y_train, train_pred))
print("Parkinson's Testing Accuracy:", accuracy_score(Y_test, test_pred))

# Save model
pickle.dump(model, open("parkinsons_model.sav", "wb"))

print("Parkinson's model saved successfully")