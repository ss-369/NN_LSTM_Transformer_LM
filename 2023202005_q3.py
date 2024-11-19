## import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm
import gensim.downloader as api

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

def load_glove_embeddings():
    print("Loading GloVe embeddings...")
    glove_model = api.load("glove-wiki-gigaword-100")  # 100-dimensional GloVe vectors
    return glove_model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2, attn_mask=mask)[0])
        x2 = self.norm2(x)
        x = x + self.dropout(self.feed_forward(x2))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        mask = self.generate_square_subsequent_mask(src.size(0)).to(src.device)
        for layer in self.layers:
            src = layer(src, mask)
        output = self.fc_out(src)
        return output

class LanguageModelDataset(Dataset):
    def __init__(self, sentences, word_to_idx, fixed_length=40, pad_idx=0):
        self.sentences = [s for s in sentences if len(s) > 0]  # Filter out empty sentences
        self.word_to_idx = word_to_idx
        self.fixed_length = fixed_length
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        # Convert words to indices
        indices = [self.word_to_idx.get(word, self.word_to_idx["<UNK>"]) for word in sentence]

        # Pad or truncate to fixed length
        if len(indices) < self.fixed_length:
            indices += [self.pad_idx] * (self.fixed_length - len(indices))  # Padding
        else:
            indices = indices[:self.fixed_length]  # Truncation

        # Prepare context and target
        context = indices[:-1]  # All but the last word
        target = indices[1:]    # All but the first word
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

def build_vocab(sentences):
    vocab = set()
    for sentence in sentences:
        vocab.update(sentence)
    vocab.add("<UNK>")  # For unknown words
    vocab.add("<PAD>")  # For padding
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return vocab, word_to_idx, idx_to_word

def sentences_to_indices(sentences, word_to_idx):
    return [[word_to_idx.get(word, word_to_idx["<UNK>"]) for word in sentence] for sentence in sentences]

def compute_perplexity(model, data_loader, criterion, device, pad_idx, output_file=None):
    model.eval()
    total_loss = 0
    batch_perplexities = []

    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(data_loader):
            src, tgt = src.to(device), tgt.to(device)
            output = model(src.transpose(0, 1))
            output = output.transpose(0, 1).contiguous().view(-1, output.size(-1))
            tgt = tgt.contiguous().view(-1)
            loss = criterion(output, tgt)
            total_loss += loss.item()
            perplexity = torch.exp(torch.tensor(loss.item())).item()
            batch_perplexities.append((batch_idx, perplexity))

    avg_loss = total_loss / len(data_loader)
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()

    if output_file:
        with open(output_file, 'w') as f:
            for batch_idx, perplexity in batch_perplexities:
                f.write(f"Batch {batch_idx}: {perplexity:.4f}\n")

    return avg_perplexity, batch_perplexities

def main():
    # Load GloVe embeddings
    glove_model = load_glove_embeddings()

    # Load data
    with open('/kaggle/working/train.txt', 'r', encoding='utf-8') as file:
        train_sentences = [line.strip().split() for line in file if line.strip()]

    with open('/kaggle/working/val.txt', 'r', encoding='utf-8') as file:
        val_sentences = [line.strip().split() for line in file if line.strip()]

    with open('/kaggle/working/test.txt', 'r', encoding='utf-8') as file:
        test_sentences = [line.strip().split() for line in file if line.strip()]

    # Build vocabulary
    vocab, word_to_idx, idx_to_word = build_vocab(train_sentences)

    # Create datasets
    fixed_length = 40  # Set the fixed length for sentences
    pad_idx = word_to_idx.get("<PAD>")  # Correctly get the padding index

    train_dataset = LanguageModelDataset(train_sentences, word_to_idx, fixed_length, pad_idx)
    val_dataset = LanguageModelDataset(val_sentences, word_to_idx, fixed_length, pad_idx)
    test_dataset = LanguageModelDataset(test_sentences, word_to_idx, fixed_length, pad_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model parameters
    vocab_size = len(vocab)
    d_model = 100
    nhead = 5
    num_layers = 3
    dim_feedforward = 512
    dropout = 0.2

    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerDecoder(vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 6
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src.transpose(0, 1))
            output = output.transpose(0, 1).contiguous().view(-1, output.size(-1))
            tgt = tgt.contiguous().view(-1)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_perplexity, _ = compute_perplexity(model, train_loader, criterion, device, pad_idx, "2023202005-LM3-train-perplexity.txt")
        val_perplexity, _ = compute_perplexity(model, val_loader, criterion, device, pad_idx, "2023202005-LM3-val-perplexity.txt")
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Train Perplexity: {train_perplexity:.4f}, Val Perplexity: {val_perplexity:.4f}")

    # Evaluate on test set
    test_perplexity, _ = compute_perplexity(model, test_loader, criterion, device, pad_idx, "2023202005-LM3-test-perplexity.txt")
    print(f"Test Perplexity: {test_perplexity:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "transformer_decoder.pth")
    print("Model saved as 'transformer_decoder.pth'")

if __name__ == "__main__":
    main()
