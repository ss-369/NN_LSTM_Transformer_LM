
---

# 2023202005 Assignment 1

## Overview

This Assignment implements various language models (NN, LSTM, Transformer Decoders) to do the language modelling. We train, validate, and test these models using perplexity as the evaluation metric.

## Files

- `2023202005-assignment1.ipynb`: Main notebook with the implementation of the models and experiments.
- `2023202005-LM1-*`: Perplexity files for NN-based language model (LM1).
- `2023202005-LM2-*`: Perplexity files for LSTM-based language model (LM2).
- `2023202005-LM3-*`: Perplexity files for Transformer Decoder-based language model (LM3).
- `2023202005_q1.py`: Python script for Question 1.
- `2023202005_q2.py`: Python script for Question 2.
- `2023202005_q3.py`: Python script for Question 3.
- `2023202005_Report.pdf`: Detailed report of the experiments.
- `tokenise.py`: Tokenization script used for preprocessing text data.
- `readme.md`: Instructions to execute the code and restore the pre-trained model.

## Setup

### Dependencies
This Assignment requires the following Python libraries:
- `torch`
- `numpy`
- `tqdm`
- `gensim`
- `matplotlib`

To install the required dependencies, run:

```bash
    pip install torch numpy tqdm gensim matplotlib

```

### Pre-trained Embeddings
We use pre-trained GloVe embeddings (100-dimensional vectors). They are automatically downloaded using `gensim` in the script.

### Dataset
The Auguste_Maquet.txt is divided into three text files:

- `train.txt`: Training data.(70%)
- `val.txt`: Validation data.(10%)
- `test.txt`: Test data.(20%)

Ensure these files are in the working directory.

### Model Training
**Training the model**:

   - Run the Jupyter notebook `2023202005-assignment1.ipynb` to train the different models. This will also save perplexity results in corresponding files for train, validation, and test sets.

   - Alternatively, you can run individual scripts (`2023202005_q1.py`, `2023202005_q2.py`, `2023202005_q3.py`) for specific questions.



### Perplexity Files
The training, validation, and test perplexities for each model are saved as:

- `2023202005-LM*-train-perplexity.txt`: Batch-wise training perplexity.
- `2023202005-LM*-val-perplexity.txt`: Batch-wise validation perplexity.
- `2023202005-LM*-test-perplexity.txt`: Batch-wise test perplexity.

### Evaluation
After training, the models are evaluated on the test set. Test perplexities for each model are saved in their respective `*-test-perplexity.txt` files.



## Code Execution

### Load Pre-trained Model

The pre-trained models are available for download [here](https://drive.google.com/drive/folders/1-oGzikyY4akBL7o51fJd2P7kftN7kFGW?usp=drive_link).




```python

    # Function to load the model
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

```
- This function loads the model's weights and optimizer's state. If the checkpoint file exists, it will load the saved epoch, model weights, and optimizer state. If not, it starts fresh.

- Example Usage:
To Restore and Continue Training: Call the load_model function before starting the training loop. It will restore the model and optimizer and resume training from the last saved epoch.

```python
    # Before training, load the pre-trained model
    start_epoch = load_model(model, optimizer, "language_model.pth")

    # Continue training from the saved epoch
    for epoch in range(start_epoch, num_epochs):
        model.train()
        # Training code follows...


```

```python
    # Load the pre-trained model
    load_model(model, optimizer, "language_model.pth")

    # Now, use the model to compute perplexity or evaluate
    test_perplexity = compute_perplexity(model, test_loader, criterion, device, "test_perplexity.txt")
    print(f"Test Perplexity: {test_perplexity:.4f}")


```


- Run the Python scripts for each model:

  ```bash
  python 2023202005_q1.py  # For NN model
  python 2023202005_q2.py  # For LSTM model
  python 2023202005_q3.py  # For Transformer model
  ```

## Results
- Perplexity scores for each model and dataset (train, validation, test) can be found in the corresponding perplexity files and 2023202005_Report.

---

