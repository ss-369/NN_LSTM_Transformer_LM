import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gensim.downloader as api
import re
from collections import Counter
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Load GloVe embeddings
def load_glove_embeddings():
    print("Loading GloVe embeddings...")
    glove_model = api.load("glove-wiki-gigaword-100")  # 100-dimensional GloVe vectors
    return glove_model

# Prepare vocabulary and index mapping
def build_vocab(sentences, glove_model):
    vocab = set()
    for sentence in sentences:
        vocab.update(sentence)

    # Add <UNK> token for unknown words
    vocab.add("<UNK>")
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    # Create an embedding matrix
    embedding_dim = glove_model.vector_size
    embedding_matrix = np.zeros((len(vocab), embedding_dim))

    for word, idx in word_to_idx.items():
        if word in glove_model:
            embedding_matrix[idx] = glove_model[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

    return vocab, word_to_idx, idx_to_word, embedding_matrix

# Convert sentences to indices
def sentences_to_indices(sentences, word_to_idx):
    return [[word_to_idx.get(word, word_to_idx["<UNK>"]) for word in sentence] for sentence in sentences]

# Define Dataset class
class LanguageModelDataset(Dataset):
    def __init__(self, sentences, context_size=5):
        self.sentences = [s for s in sentences if len(s) >= context_size + 1]
        self.context_size = context_size

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        context = sentence[:self.context_size]
        target = sentence[self.context_size]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class NeuralLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_matrix):
        super(NeuralLanguageModel, self).__init__()

        # Use pre-trained GloVe embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embeddings.weight.requires_grad = False  # Keep the embeddings fixed (non-trainable)

        self.fc1 = nn.Linear(embedding_dim * 5, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_dim, vocab_size)  # Output layer for vocabulary size
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer with 50% rate
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embeddings(x)  # Shape: (batch_size, context_size, embedding_dim)
        x = x.view(x.size(0), -1)  # Flatten the embeddings
        x = torch.relu(self.fc1(x))  # First hidden layer
        x = torch.relu(self.fc2(x))  # Second hidden layer
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)  # Output layer
        x = self.softmax(x)
        return x

# Compute perplexity and write to file
def compute_perplexity(model, data_loader, criterion, device, file_name):
    model.eval()
    total_loss = 0
    total_words = 0
    batch_num = 0

    with open(file_name, "w") as f:
        with torch.no_grad():
            for context, target in data_loader:
                batch_num += 1
                if context is None:
                    continue
                context, target = context.to(device), target.to(device)
                output = model(context)
                loss = criterion(output, target)
                total_loss += loss.item() * context.size(0)
                total_words += context.size(0)

                avg_loss = loss.item()
                perplexity = np.exp(avg_loss)
                f.write(f"Batch {batch_num}: Perplexity = {perplexity:.4f}\n")

    avg_loss = total_loss / total_words
    perplexity = np.exp(avg_loss)
    return perplexity

# Main function
def main():
    # Load data
    def load_and_preprocess(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            sentences = [line.strip().split() for line in file if line.strip()]  # Don't remove punctuation
        return sentences

    train_sentences = load_and_preprocess('/kaggle/working/train.txt')
    val_sentences = load_and_preprocess('/kaggle/working/val.txt')
    test_sentences = load_and_preprocess('/kaggle/working/test.txt')

    # Load embeddings
    glove_model = load_glove_embeddings()
    vocab, word_to_idx, idx_to_word, embedding_matrix = build_vocab(train_sentences, glove_model)

    # Convert sentences to indices
    train_indices = sentences_to_indices(train_sentences, word_to_idx)
    val_indices = sentences_to_indices(val_sentences, word_to_idx)
    test_indices = sentences_to_indices(test_sentences, word_to_idx)

    # Create datasets and dataloaders
    train_dataset = LanguageModelDataset(train_indices)
    val_dataset = LanguageModelDataset(val_indices)
    test_dataset = LanguageModelDataset(test_indices)

    # Check if datasets are empty
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Check the input data.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty. Check the input data.")
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty. Check the input data.")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralLanguageModel(vocab_size=len(vocab), embedding_dim=100, hidden_dim=300, embedding_matrix=embedding_matrix).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Train the model
    num_epochs = 6
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for context, target in tqdm(train_loader):
            context, target = context.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * context.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Compute and print perplexity on training set
        train_perplexity = compute_perplexity(model, train_loader, criterion, device, "2023202005-LM1-train-perplexity.txt")
        print(f"Training Perplexity: {train_perplexity:.4f}")

        # Compute and print perplexity on validation set
        val_perplexity = compute_perplexity(model, val_loader, criterion, device, "2023202005-LM1-val-perplexity.txt")
        print(f"Validation Perplexity: {val_perplexity:.4f}")

    # Compute and print perplexity on test set
    test_perplexity = compute_perplexity(model, test_loader, criterion, device, "2023202005-LM1-test-perplexity.txt")
    print(f"Test Perplexity: {test_perplexity:.4f}")

if __name__ == "__main__":
    main()
