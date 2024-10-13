
---

# Transformer Decoder-based Language Model

## Description

This project implements a **Transformer Decoder-based Language Model** using PyTorch. The model predicts the next word in a sequence, relying on the architecture introduced in **"Attention is All You Need"**. The model is trained on the **Auguste Maquet** corpus, and its performance is evaluated using **perplexity scores**.

## Model Architecture

The Transformer Decoder-based Language Model consists of:
1. **Embedding Layer**: Converts input tokens into dense vector representations.
2. **Positional Encoding**: Adds information about the position of tokens in the sequence.
3. **Transformer Decoder Layers**: Multiple stacked decoder layers with self-attention and feedforward layers.
4. **Output Layer**: A softmax layer that predicts the probability distribution of the next word in the vocabulary.

### Key Components:
- **Self-attention Mechanism**: Captures dependencies between tokens in the input sequence.
- **Feedforward Network**: Applies transformations to the attention output.

## Dataset

- **Corpus**: Auguste Maquet corpus (as provided in the assignment).
- **Preprocessing**: The corpus is cleaned to remove special characters and tokenized for training.
- **Train, Validation, and Test Splits**:
  - 10,000 sentences for validation.
  - 20,000 sentences for testing.

## Evaluation Metric

- **Perplexity**: Used as the primary evaluation metric to measure the language model's performance on both the train and test datasets.

## How to Run

### 1. Install Dependencies
Ensure that PyTorch and other required libraries are installed:
```bash
pip install torch
```

### 2. Train the Transformer Decoder-based Language Model
To train the model, run the following command:
```bash
python transformer_decoder_lm.py
```
The model will begin training using the Auguste Maquet corpus. Perplexity scores will be generated and saved after each epoch.

### 3. Evaluate the Model
After training, the model can be evaluated on the validation and test datasets to compute the perplexity scores:
```bash
python transformer_decoder_lm.py --evaluate
```

### 4. Pretrained Models
If the pretrained models exceed the file size limit, upload them to OneDrive or Google Drive and include the link here in the `README.md`. You can load the pretrained model by running:
```bash
python transformer_decoder_lm.py --load_model <path_to_model>
```

## Perplexity Files

The perplexity scores will be saved in text files with the following format:
- **Train Perplexity**: `rollnumber_LM3_train_perplexity.txt`
- **Test Perplexity**: `rollnumber_LM3_test_perplexity.txt`

Each file will contain one line per sentence in the format:
```
Sentence TAB perplexity-score
```
At the end of the file, the average perplexity score will be reported.

## Hyperparameter Tuning

- Experiment with different numbers of **Transformer decoder layers**, **dropout rates**, and **learning rates**.
- Report the results in the accompanying PDF file and compare perplexity scores across different configurations.

## Submission Format

Zip the following files into one archive:
1. **Source Code**:
   - `transformer_decoder_lm.py`
2. **Perplexity Files**:
   - `rollnumber_LM3_train_perplexity.txt`
   - `rollnumber_LM3_test_perplexity.txt`
3. **Pretrained Models**:
   - `transformer_decoder_model.pt` (if applicable, provide a link for large files).
4. **README.md**: Include instructions on how to execute the code and load pretrained models.

Name the zip file as `<roll_number>_assignment1.zip`.

## Resources
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

---

