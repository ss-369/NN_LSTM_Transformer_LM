import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gensim.downloader as api
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

def load_glove_embeddings():
    print("Loading GloVe embeddings...")
    glove_model = api.load("glove-wiki-gigaword-100")  # 100-dimensional GloVe vectors
    return glove_model

def build_vocab(sentences, glove_model):
    vocab = set()
    for sentence in sentences:
        vocab.update(sentence)

    vocab.add("<PAD>")
    vocab.add("<UNK>")
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    embedding_dim = glove_model.vector_size
    embedding_matrix = np.zeros((len(vocab), embedding_dim))

    for word, idx in word_to_idx.items():
        if word in glove_model:
            embedding_matrix[idx] = glove_model[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

    return vocab, word_to_idx, idx_to_word, embedding_matrix

class LanguageModelDataset(Dataset):
    def __init__(self, sentences, word_to_idx, seq_length=40):
        self.sentences = sentences
        self.word_to_idx = word_to_idx
        self.seq_length = seq_length
        self.data = self.prepare_data()

    def prepare_data(self):
        data = []
        pad_idx = self.word_to_idx["<PAD>"]
        for sentence in self.sentences:
            if len(sentence) > 1:  # Ensure the sentence has at least 2 words
                indexed_sentence = [self.word_to_idx.get(word, self.word_to_idx["<UNK>"]) for word in sentence]

                # Pad or truncate the sentence to seq_length + 1
                if len(indexed_sentence) < self.seq_length + 1:
                    indexed_sentence = indexed_sentence + [pad_idx] * (self.seq_length + 1 - len(indexed_sentence))
                else:
                    indexed_sentence = indexed_sentence[:self.seq_length + 1]

                # Create input-target pairs
                seq = indexed_sentence[:self.seq_length]
                target = indexed_sentence[1:self.seq_length + 1]

                data.append((seq, target))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, target = self.data[idx]
        return torch.tensor(seq), torch.tensor(target)

class NeuralLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_matrix):
        super(NeuralLanguageModel, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embeddings.weight.requires_grad = False

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.embeddings(x)  # Shape: (batch_size, seq_length, embedding_dim)
        x, _ = self.lstm(x)  # Shape: (batch_size, seq_length, hidden_dim)
        x = self.dropout(x)
        x = self.fc(x)  # Shape: (batch_size, seq_length, vocab_size)
        return x

def compute_perplexity(model, data_loader, criterion, device, file_name, is_train=False):
    model.eval()
    total_loss = 0
    total_words = 0
    batch_perplexities = []

    with open(file_name, "w") as f:
        with torch.no_grad():
            for batch_num, (seq, target) in enumerate(data_loader, 1):
                seq, target = seq.to(device), target.to(device)
                output = model(seq)
                loss = criterion(output.view(-1, output.size(-1)), target.view(-1))

                non_pad_elements = (target != 0).sum().item()
                batch_loss = loss.item() * non_pad_elements
                total_loss += batch_loss
                total_words += non_pad_elements

                batch_perplexity = np.exp(batch_loss / non_pad_elements)
                batch_perplexities.append(batch_perplexity)

                if is_train:
                    f.write(f"Batch {batch_num}: Perplexity = {batch_perplexity:.4f}\n")
                else:
                    f.write(f"Batch {batch_num}: Perplexity = {batch_perplexity:.4f}\n")

        if is_train:
            avg_perplexity = np.mean(batch_perplexities)
            f.write(f"\nAverage Perplexity: {avg_perplexity:.4f}\n")
            return avg_perplexity
        else:
            avg_loss = total_loss / total_words
            avg_perplexity = np.exp(avg_loss)
            return avg_perplexity

# Function to save the model
def save_model(model, epoch, optimizer, path="LSTM_language_model.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Model saved after epoch {epoch} at {path}")

def main():
    def load_and_preprocess(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            sentences = [line.strip().split() for line in file if line.strip()]
        return sentences

    train_sentences = load_and_preprocess('/kaggle/working/train.txt')
    val_sentences = load_and_preprocess('/kaggle/working/val.txt')
    test_sentences = load_and_preprocess('/kaggle/working/test.txt')

    glove_model = load_glove_embeddings()
    vocab, word_to_idx, idx_to_word, embedding_matrix = build_vocab(train_sentences, glove_model)

    train_dataset = LanguageModelDataset(train_sentences, word_to_idx)
    val_dataset = LanguageModelDataset(val_sentences, word_to_idx)
    test_dataset = LanguageModelDataset(test_sentences, word_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralLanguageModel(vocab_size=len(vocab), embedding_dim=100, hidden_dim=300, embedding_matrix=embedding_matrix).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for seq, target in tqdm(train_loader):
            seq, target = seq.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * seq.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        save_model(model, epoch + 1, optimizer)


        # Compute and print perplexity for training set after each epoch
        train_perplexity = compute_perplexity(model, train_loader, criterion, device, "2023202005-LM2-train-perplexity.txt", is_train=True)
        print(f"Epoch {epoch + 1} Training Perplexity: {train_perplexity:.4f}")

        # Compute perplexity for validation set after each epoch
        val_perplexity = compute_perplexity(model, val_loader, criterion, device, "2023202005-LM2-val-perplexity.txt")
        print(f"Epoch {epoch + 1} Validation Perplexity: {val_perplexity:.4f}")

    # Compute perplexity for test set after training
    test_perplexity = compute_perplexity(model, test_loader, criterion, device, "2023202005-LM2-test-perplexity.txt")
    print(f"Final Test Perplexity: {test_perplexity:.4f}")

if __name__ == "__main__":
    main()
