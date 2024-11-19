---

# **2023202005 Assignment 1**

## **Overview**

This assignment involves implementing and evaluating three language models: 
1. **NN-based Language Model (LM1)**  
2. **LSTM-based Language Model (LM2)**  
3. **Transformer Decoder-based Language Model (LM3)**  

The models are trained, validated, and tested on the **Auguste Maquet corpus**, with **perplexity** as the evaluation metric. Each model's performance is analyzed and compared.

---

## **Files**

- `2023202005-assignment1.ipynb`: Main notebook for implementing and running experiments.
- `2023202005-LM1-*`: Perplexity files for NN-based language model.
- `2023202005-LM2-*`: Perplexity files for LSTM-based language model.
- `2023202005-LM3-*`: Perplexity files for Transformer Decoder-based language model.
- `2023202005_q1.py`: Python script for NN-based language model.
- `2023202005_q2.py`: Python script for LSTM-based language model.
- `2023202005_q3.py`: Python script for Transformer Decoder-based language model.
- `2023202005_Report.pdf`: Report containing experiment details, results, and analysis.
- `tokenise.py`: Preprocessing script for tokenizing text data.
- `readme.md`: Detailed instructions for running the code and loading pretrained models.

---

## **Setup**

### **Dependencies**

Install the required Python libraries:
```bash
  pip install torch numpy tqdm gensim matplotlib
```

### **Pre-trained Embeddings**

Pre-trained **GloVe embeddings** (100-dimensional) are used. The embeddings are automatically downloaded using the `gensim` library during training.

### **Dataset**

The **Auguste Maquet corpus** is split into three parts:
- `train.txt` (70%): Training data.
- `val.txt` (10%): Validation data.
- `test.txt` (20%): Test data.

Ensure these files are in the working directory.

---

## **Model Training and Evaluation**

### **Training**

To train the models, use the Jupyter notebook `2023202005-assignment1.ipynb`. This will train the NN, LSTM, and Transformer models sequentially and save the perplexity results for each model.

Alternatively, use the individual scripts:
- For NN-based model: `python 2023202005_q1.py`
- For LSTM-based model: `python 2023202005_q2.py`
- For Transformer Decoder-based model: `python 2023202005_q3.py`

### **Perplexity Files**

Perplexity scores for train, validation, and test sets are saved in text files:
- **NN-based LM (LM1)**:
  - `2023202005-LM1-train-perplexity.txt`
  - `2023202005-LM1-val-perplexity.txt`
  - `2023202005-LM1-test-perplexity.txt`
- **LSTM-based LM (LM2)**:
  - `2023202005-LM2-train-perplexity.txt`
  - `2023202005-LM2-val-perplexity.txt`
  - `2023202005-LM2-test-perplexity.txt`
- **Transformer-based LM (LM3)**:
  - `2023202005-LM3-train-perplexity.txt`
  - `2023202005-LM3-val-perplexity.txt`
  - `2023202005-LM3-test-perplexity.txt`

Each file contains line-wise perplexity scores for sentences, and the average perplexity is reported at the end.

### **Evaluation**

The models are evaluated using perplexity on the test set. The test perplexities for each model are saved in their respective files.

---

## **Loading Pre-trained Models**

Pre-trained models can be downloaded from [this link](https://drive.google.com/drive/folders/1-oGzikyY4akBL7o51fJd2P7kftN7kFGW?usp=drive_link).  

To load and evaluate a pre-trained model:
```python
  # Function to load model
  def load_model(model, optimizer, path="language_model.pth"):
      if os.path.isfile(path):
          checkpoint = torch.load(path)
          model.load_state_dict(checkpoint['model_state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
          epoch = checkpoint['epoch']
          print(f"Model loaded from {path}, resuming from epoch {epoch}")
          return epoch
      else:
          print(f"No checkpoint found at {path}")
          return 0
  
  # Example usage
  start_epoch = load_model(model, optimizer, "language_model.pth")
```

### Usage for Evaluation:
```python
  # Load pre-trained model
  load_model(model, optimizer, "language_model.pth")
  
  # Compute perplexity on test set
  test_perplexity = compute_perplexity(model, test_loader, criterion, device, "test_perplexity.txt")
  print(f"Test Perplexity: {test_perplexity:.4f}")
```

---

## **Submission Format**

Submit a zip file named as `<roll_number>_assignment1.zip`, containing:
1. **Source Code**:
   - `2023202005-assignment1.ipynb`
   - `2023202005_q1.py`, `2023202005_q2.py`, `2023202005_q3.py`
   - `tokenise.py`
2. **Perplexity Files**:
   - `2023202005-LM1-*`
   - `2023202005-LM2-*`
   - `2023202005-LM3-*`
3. **Pretrained Models**:
   - Include a download link for large files if needed.
4. **README.md**:
   - Instructions to execute the code and load pre-trained models.

---

## **Results**

The detailed analysis, results, and comparisons of perplexity scores for each model can be found in `2023202005_Report.pdf`.

---

## **Resources**

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

--- 

