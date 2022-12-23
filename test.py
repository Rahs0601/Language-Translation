import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the translation data
df = pd.read_csv("eng-kannada.csv")

# Split the data into features (X) and labels (y)
# X = df[["English"]]
# y = df["Kannada"]
from sklearn.feature_extraction.text import CountVectorizer

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Fit the vectorizer on the English and Kannada sentences
vectorizer.fit(df["English"] + df["Kannada"])

# Transform the English and Kannada sentences into numerical vectors
X = vectorizer.transform(df["English"])
y = vectorizer.transform(df["Kannada"])
# Convert the sparse matrices to dense arrays
X = X.toarray()
y = y.toarray()


# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a random forest classifier on the training data
clf = RandomForestClassifier()
clf.fit(X_train[1000], y_train[1000])
# print(X_train)
# Make predictions on the test data
predictions = clf.predict(X_test[1000])

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Translate a sentence from English to Kannada
english_sentence = "Hello, how are you?"
kannada_translation = clf.predict([[english_sentence]])[0]
print("English:", english_sentence)
print("Kannada:", kannada_translation)

# Translate a sentence from Kannada to English
kannada_sentence = "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗೆ ಹೋಗುತ್ತೀರಿ?"
english_translation = clf.predict([[kannada_sentence]])[0]
print("Kannada:", kannada_sentence)
print("English:", english_translation)
