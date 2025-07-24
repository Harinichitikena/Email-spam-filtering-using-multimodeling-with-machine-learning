import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
mail = pd.read_csv(r"/content/Spam_Data.csv")
mail.dropna(inplace=True)  # Remove missing values
mail["filter"] = mail["Category"].map({"ham": 0, "spam": 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    mail['Message'], mail['filter'], train_size=0.8, random_state=42)

# Vectorization (Convert text to numerical representation)
v = CountVectorizer()
X_train_count = v.fit_transform(X_train)
X_test_count = v.transform(X_test)

# Models to evaluate
models = {
    "Naïve Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear')
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train_count, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train_count))
    test_acc = accuracy_score(y_test, model.predict(X_test_count))
    results[name] = (train_acc, test_acc)

# Print results
print("\nModel Performance Comparison:")
best_model = None
best_test_acc = 0

train_accuracies = []
test_accuracies = []
model_names = []

for name, (train_acc, test_acc) in results.items():
    print(f"{name}: Train Accuracy = {train_acc * 100:.2f}%, Test Accuracy = {test_acc * 100:.2f}%")
    model_names.append(name)
    train_accuracies.append(train_acc * 100)
    test_accuracies.append(test_acc * 100)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_model = name

# Show the best model
print(f"\nBest Model: {best_model} with Test Accuracy = {best_test_acc * 100:.2f}%")

# Plot accuracies
plt.figure(figsize=(8, 5))
bar_width = 0.35
index = np.arange(len(model_names))
plt.bar(index, train_accuracies, bar_width, label='Train Accuracy', alpha=0.7)
plt.bar(index + bar_width, test_accuracies, bar_width, label='Test Accuracy', alpha=0.7)
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Model Comparison')
plt.xticks(index + bar_width / 2, model_names, rotation=15)
plt.legend()
plt.show()

# Predict on sample emails
emails = [
    'Hey Mohamed, can we get together to watch football game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!',
    'You have a big offer in many markets and you have 20% cash back ',
    'I have been searching for the right words to thank you for this breather. You have been wonderful.'
]
emails_count = v.transform(emails)

# Predictions from the best model
print(f"\nPredictions using the Best Model ({best_model}):")
best_model_instance = models[best_model]
prediction = best_model_instance.predict(emails_count)
for i, pred in enumerate(prediction):
    label = "Spam" if pred == 1 else "Ham"
    print(f"Email {i+1}: {label}")