---

# **NN_LSTM_Transformer_LM**

This repository contains the implementation and evaluation of three language models:  
1. **NN-based Language Model (LM1)**  
2. **LSTM-based Language Model (LM2)**  
3. **Transformer Decoder-based Language Model (LM3)**  

The models are trained on the **Auguste Maquet corpus** and evaluated using **perplexity** as the primary metric. The repository provides scripts for training, validating, and testing the models, along with detailed results and analysis.

---

## **Overview**

### Models Implemented:
- **NN-based Language Model (LM1)**: A simple neural network-based model.  
- **LSTM-based Language Model (LM2)**: A recurrent model leveraging Long Short-Term Memory units.  
- **Transformer Decoder-based Language Model (LM3)**: An attention-based model derived from the "Attention is All You Need" architecture.

### Dataset:
- The **Auguste Maquet corpus** is used for training and evaluation.  
- The data is split into:
  - **70%** for training
  - **10%** for validation
  - **20%** for testing.

### Evaluation:
- **Perplexity** is used to measure model performance on both the validation and test sets.

---

## **Repository Structure**

```
NN_LSTM_Transformer_LM/
├── 2023202005-assignment1.ipynb  # Main notebook for all models
├── 2023202005_q1.py             # NN-based Language Model script
├── 2023202005_q2.py             # LSTM-based Language Model script
├── 2023202005_q3.py             # Transformer Decoder-based Language Model script
├── tokenise.py                  # Script for preprocessing and tokenization
├── train.txt                    # Training dataset
├── val.txt                      # Validation dataset
├── test.txt                     # Testing dataset
├── pretrained_models/           # Directory for pre-trained models
│   └── transformer_model.pth    # Example: Pre-trained Transformer model
├── results/                     # Directory for perplexity results
│   ├── LM1-train-perplexity.txt
│   ├── LM1-val-perplexity.txt
│   ├── LM1-test-perplexity.txt
│   ├── LM2-*.txt
│   └── LM3-*.txt
├── 2023202005_Report.pdf        # Report with experiment details
├── README.md                    # Repository documentation (this file)
```

---

## **Setup**

### Dependencies

Install the following Python libraries:
```bash
pip install torch numpy tqdm gensim matplotlib
```

### Dataset

Ensure that `train.txt`, `val.txt`, and `test.txt` are present in the working directory.

---

## **Usage**

### Training the Models

Run the following scripts to train individual models:
- **NN-based model**:
  ```bash
  python 2023202005_q1.py
  ```
- **LSTM-based model**:
  ```bash
  python 2023202005_q2.py
  ```
- **Transformer Decoder-based model**:
  ```bash
  python 2023202005_q3.py
  ```

Alternatively, use the notebook `2023202005-assignment1.ipynb` to train and evaluate all models.

### Pre-trained Models

Pre-trained models are available for download in the `pretrained_models` directory. You can also download them from this [link](https://drive.google.com/drive/folders/1-oGzikyY4akBL7o51fJd2P7kftN7kFGW?usp=drive_link).

To load a pre-trained model:
```python
# Function to load a pre-trained model
def load_model(model, optimizer, path="pretrained_models/transformer_model.pth"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model loaded successfully!")
```

### Evaluating the Models

After training, perplexity scores are saved in the `results` directory. Evaluate the test perplexity as follows:
```python
test_perplexity = compute_perplexity(model, test_loader, criterion, device)
print(f"Test Perplexity: {test_perplexity:.4f}")
```

---

## **Results**

The perplexity scores for each model are saved in the `results` directory:
- `LM1-*`: NN-based language model results.
- `LM2-*`: LSTM-based language model results.
- `LM3-*`: Transformer Decoder-based language model results.

Detailed analysis and comparisons are included in `2023202005_Report.pdf`.

---

## **Submission**

Include the following in your submission:
1. Source code (`*.py` and `*.ipynb` files).
2. Perplexity results (`results` directory).
3. Pre-trained models (or a download link if the files are large).
4. A detailed report (`2023202005_Report.pdf`).

Compress these into a zip file named:  
`2023202005_Assignment1.zip`.

---

## **References**

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
2. [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
3. [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)  

--- 

