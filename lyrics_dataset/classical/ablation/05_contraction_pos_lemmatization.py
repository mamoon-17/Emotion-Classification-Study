from datasets import load_dataset
import pandas as pd
import re
import nltk
import contractions
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
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
df = pd.DataFrame(data)[['lyrics', 'mood']]

# ========== PREPROCESSING ==========
print("🧹 Cleaning lyrics...")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'}
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN,
                "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_text(text):
    # Expand contractions: can't -> cannot
    text = contractions.fix(text)
    # Remove things like [Chorus] or [Verse]
    text = re.sub(r'\[.*?\]', '', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove repeated letters (soooo -> soo)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = text.lower().strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t, get_wordnet_pos(t))
              for t in tokens if t not in stop_words]
    return ' '.join(tokens)

df['clean_lyrics'] = df['lyrics'].apply(preprocess_text)

# ========== TRAIN/TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_lyrics'], df['mood'],
    test_size=0.2, random_state=42, stratify=df['mood']
)

# ========== CLASS WEIGHTS ==========
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

# ========== PIPELINE & GRID SEARCH ==========
print("🔍 Running GridSearchCV...")

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(sublinear_tf=True)),
    ('logreg', LogisticRegression(max_iter=3000, class_weight=class_weights))
])

params = {
    'tfidf__max_features': [20000, 30000],
    'tfidf__ngram_range': [(1,2), (1,3)],
    'tfidf__analyzer': ['word', 'char_wb'],   # add char n-grams
    'tfidf__min_df': [2, 3],
    'tfidf__max_df': [0.8, 0.9],
    'logreg__C': [1, 10, 50, 100],
    'logreg__penalty': ['l2'],
    'logreg__solver': ['lbfgs', 'saga']
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
