from datasets import load_dataset
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# ========== LOAD DATASET ==========
print("📥 Loading dataset...")
dataset = load_dataset("Annanay/aml_song_lyrics_balanced")
data = dataset['train']
df = pd.DataFrame(data)[['lyrics', 'mood', 'mood_cats']]

# ========== PREPROCESSING ==========
print("🧹 Cleaning lyrics...")
nltk.download('stopwords')
nltk.download('wordnet')

# Keep negation words for sentiment
stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'}
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'\[.*?\]', '', text)          # remove brackets
    text = re.sub(r'[^a-zA-Z\s]', '', text)      # keep letters only
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

df['clean_lyrics'] = df['lyrics'].apply(preprocess_text)

# ========== TRAIN/TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_lyrics'], df['mood'],
    test_size=0.2, random_state=42
)

# ========== CLASS WEIGHTS ==========
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

# ========== GRID SEARCH PIPELINE ==========
print("🔍 Running GridSearchCV...")

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(sublinear_tf=True)),
    ('logreg', LogisticRegression(max_iter=2000, class_weight=class_weights))
])

params = {
    'tfidf__max_features': [15000, 20000],
    'tfidf__ngram_range': [(1,2), (1,3)],
    'tfidf__min_df': [2, 3],
    'tfidf__max_df': [0.8],
    'logreg__C': [10, 20, 50]
}

grid = GridSearchCV(pipeline, param_grid=params, cv=3, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

print("\n✅ Grid search complete!")
print("Best params:", grid.best_params_)
print("Best cross-val accuracy:", round(grid.best_score_, 4))

# ========== EVALUATION ==========
print("\n🚀 Evaluating on test set...")
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("✅ Model trained successfully!")
print("Test Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))