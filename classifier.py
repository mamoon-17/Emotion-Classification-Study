from datasets import load_dataset
import pandas as pd

dataset = load_dataset("Annanay/aml_song_lyrics_balanced")

#print(dataset)
data = dataset['train']

df = pd.DataFrame(data)

df = df[['lyrics', 'mood', 'mood_cats']]
print(df.head())