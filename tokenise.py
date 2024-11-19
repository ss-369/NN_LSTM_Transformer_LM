import re
from nltk.tokenize import sent_tokenize, word_tokenize
import random

import nltk
nltk.download('punkt')

def clean_text_for_nlp(text):
    text = re.sub(r'the project gutenberg ebook.*?contents', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-z0-9.,!?;\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_sentences(sentences):
    processed = []
    for sentence in sentences:
        sentence = re.sub(r'([.,!?;])', r' \1', sentence)
        words = word_tokenize(sentence)
        if len(words) > 5:
            processed.append(sentence)
    return processed

with open('/kaggle/input/data-set/Auguste_Maquet.txt', 'r', encoding='utf-8') as file:
    text = file.read()

print(f"Original text length: {len(text)} characters")

cleaned_text = clean_text_for_nlp(text)

print(f"Cleaned text length: {len(cleaned_text)} characters")

sentences = sent_tokenize(cleaned_text)

print(f"Number of sentences after initial tokenization: {len(sentences)}")

cleaned_sentences = process_sentences(sentences)

print(f"Total number of sentences after cleaning: {len(cleaned_sentences)}")

print("\nFirst 5 cleaned sentences:")
for i, sentence in enumerate(cleaned_sentences[:5]):
    print(f"{i+1}. {sentence}")

random.shuffle(cleaned_sentences)

train_size = int(0.7 * len(cleaned_sentences))
val_size = int(0.1 * len(cleaned_sentences))
test_size = len(cleaned_sentences) - train_size - val_size

train_sentences = cleaned_sentences[:train_size]
val_sentences = cleaned_sentences[train_size:train_size+val_size]
test_sentences = cleaned_sentences[train_size+val_size:]

def write_to_file(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data))

write_to_file('train.txt', train_sentences)
write_to_file('val.txt', val_sentences)
write_to_file('test.txt', test_sentences)

print(f"\nData split complete:")
print(f"Train set: {len(train_sentences)} sentences")
print(f"Validation set: {len(val_sentences)} sentences")
