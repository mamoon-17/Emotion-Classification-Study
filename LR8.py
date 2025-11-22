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
from textblob import TextBlob

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
    text = contractions.fix(text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    # Keep ! and ? since they convey emotion
    text = re.sub(r'[^a-zA-Z!?\'\s]', '', text)
    text = re.sub(r'(.)\1{4,}', r'\1\1', text)  # keep mild repetition
    text = text.lower().strip()
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    cleaned = [
        lemmatizer.lemmatize(t, get_wordnet_pos(tag))
        for t, tag in tagged if t not in stop_words
    ]
    return ' '.join(cleaned)

df['clean_lyrics'] = df['lyrics'].apply(preprocess_text)

# Extra sentiment features
print("💡 Adding sentiment features...")
def extract_features(text):
    blob = TextBlob(text)
    return pd.Series({
        'sentiment': blob.sentiment.polarity,
        'exclamations': text.count('!'),
        'questions': text.count('?'),
        'first_person': len(re.findall(r'\b(i|me|my|mine)\b', text))
    })

extra_features = df['lyrics'].apply(extract_features)
df = pd.concat([df, extra_features], axis=1)

# ========== TRAIN/TEST SPLIT ==========
X_train_text, X_test_text, X_train_meta, X_test_meta, y_train, y_test = train_test_split(
    df['clean_lyrics'],
    df[['sentiment', 'exclamations', 'questions', 'first_person']],
    df['mood'],
    test_size=0.2,
    random_state=42,
    stratify=df['mood']
)

# ========== CLASS WEIGHTS ==========
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

# ========== FEATURE UNION (word + char n-grams) ==========
tfidf_word = TfidfVectorizer(
    analyzer='word', ngram_range=(1, 3),
    min_df=2, max_df=0.9, max_features=50000,
    sublinear_tf=True, smooth_idf=True
)

tfidf_char = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(3, 5),
    min_df=2, max_df=0.9, max_features=30000,
    sublinear_tf=True, smooth_idf=True
)

combined_features = FeatureUnion([
    ('word_tfidf', tfidf_word),
    ('char_tfidf', tfidf_char)
])

# ========== PIPELINE ==========
pipeline = Pipeline([
    ('features', combined_features),
    ('logreg', LogisticRegression(
        max_iter=4000,
        class_weight=class_weights,
        solver='saga',
        C=10
    ))
])

# ========== TRAINING ==========
print("🚀 Training Logistic Regression...")
start = time.time()
pipeline.fit(X_train_text, y_train)
end = time.time()
print(f"✅ Training complete in {round((end-start)/60,2)} minutes.")

# ========== EVALUATION ==========
print("\n🔍 Evaluating on test set...")
y_pred = pipeline.predict(X_test_text)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Macro F1 Score:", round(f1_score(y_test, y_pred, average='macro'), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ========== CONFUSION MATRIX ==========
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap='Purples', xticks_rotation=45)
plt.title("Confusion Matrix - Improved Logistic Regression")
plt.show()
