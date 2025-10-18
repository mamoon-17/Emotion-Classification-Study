from datasets import load_dataset
import pandas as pd
import re
import nltk
import contractions
import time
import numpy as np

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

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
nltk.download('punkt')

stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'}
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    tag = tag[0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN,
                "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_text(text):
    # Basic cleaning
    text = contractions.fix(text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z!?\'\s]', '', text)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)  # limit repeated chars
    text = text.lower().strip()
    
    # Tokenization and Lemmatization
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    cleaned = [
        lemmatizer.lemmatize(t, get_wordnet_pos(tag))
        for t, tag in tagged if t not in stop_words
    ]
    return ' '.join(cleaned)

# Apply preprocessing
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

# ========== PIPELINE ==========
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(sublinear_tf=True)),
    ('logreg', LogisticRegression(max_iter=3000, class_weight=class_weights))
])

# ========== RANDOMIZED SEARCH ==========
params = {
    'tfidf__max_features': [20000, 30000, 40000],
    'tfidf__ngram_range': [(1, 2), (1, 3)],
    'tfidf__min_df': [2, 3],
    'tfidf__max_df': [0.8, 0.9],
    'tfidf__analyzer': ['word', 'char_wb'],
    'logreg__C': np.logspace(-2, 2, 10),
    'logreg__solver': ['lbfgs', 'saga']
}

print("🔍 Running RandomizedSearchCV...")
start = time.time()
search = RandomizedSearchCV(
    pipeline,
    param_distributions=params,
    n_iter=25,  # faster than full grid search
    cv=3,
    n_jobs=-1,
    verbose=2,
    random_state=42
)
search.fit(X_train, y_train)
end = time.time()

print(f"\n✅ Randomized search complete! (took {round((end - start)/60, 2)} min)")
print("Best params:", search.best_params_)
print("Best cross-val accuracy:", round(search.best_score_, 4))

# ========== EVALUATION ==========
print("\n🚀 Evaluating on test set...")
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print("✅ Model trained successfully!")
print("Test Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Macro F1 Score:", round(f1_score(y_test, y_pred, average='macro'), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ========== CONFUSION MATRIX ==========
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - Logistic Regression")
plt.show()
