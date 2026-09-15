import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import os

print("Loading data...")
# Load the dataset
df = pd.read_csv('data/raw/Phishing_Email.csv')

# Remove rows with missing email text
df = df.dropna(subset=['Email Text'])

# Convert to string (in case of mixed types)
df['Email Text'] = df['Email Text'].astype(str)

# Convert Email Type to binary (Safe=0, Phishing=1)
df['is_phishing'] = (df['Email Type'] == 'Phishing Email').astype(int)

print(f"Total emails (after cleaning): {len(df)}")
print(f"Phishing: {(df['is_phishing'] == 1).sum()}")
print(f"Safe: {(df['is_phishing'] == 0).sum()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['Email Text'], df['is_phishing'], test_size=0.2, random_state=42
)

print("\nVectorizing text...")
# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

print("Training model...")
# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_transformed, y_train)

# Evaluate
y_pred = model.predict(X_test_transformed)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print("="*50)

# Save model and vectorizer
os.makedirs('models', exist_ok=True)
pickle.dump(model, open('models/phishing_model.pkl', 'wb'))
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))

print()