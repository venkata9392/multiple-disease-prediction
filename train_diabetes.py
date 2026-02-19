import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("D:\\multiple\\diabetes.csv")

# Features and Target
X = data.drop("Outcome", axis=1)
Y = data["Outcome"]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2)

# Train model
model = SVC(kernel='linear')
model.fit(X_train, Y_train)

# Accuracy
X_train_pred = model.predict(X_train)
train_acc = accuracy_score(Y_train, X_train_pred)

X_test_pred = model.predict(X_test)
test_acc = accuracy_score(Y_test, X_test_pred)

print("Diabetes Training Accuracy:", train_acc)
print("Diabetes Testing Accuracy:", test_acc)

# Save model
pickle.dump(model, open("diabetes_model.sav", "wb"))

print("Diabetes model saved successfully")
