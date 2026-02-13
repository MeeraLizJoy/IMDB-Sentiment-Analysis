# Sentiment Analysis on IMDB Dataset

## Overview
This project involves developing deep learning models for binary sentiment classification of movie reviews from the IMDB dataset. The objective was to compare various architectures, optimize hyperparameters, and explore advanced models like transformers.

---

## Models Explored

### 1. Dense Neural Network
- A basic feed-forward neural network with embedding layer followed by dense layers.
- Used as a baseline for performance comparison.
- Achieved approximately 85.8% accuracy on test data.

### 2. Bidirectional LSTM
- Captures sequential dependencies in both forward and backward directions.
- Suitable for understanding context in text.
- Achieved around 84.1% accuracy initially, improved to ~85.6% after hyperparameter tuning.

### 3. Convolutional Neural Network (CNN)
- Uses convolutional filters to detect local features in text.
- Outperformed other models in baseline accuracy (~86.1%).
- Selected as the best at initial stage before tuning.

---

## Hyperparameter Tuning
- Used **Keras Tuner** to optimize the LSTM model:
  - Tuned parameters such as embedding size, LSTM units, dropout rate, and learning rate.
  - Conducted 10 trials, selecting the best configuration.
  - Best hyperparameters found:
    - Embedding dimension: 128
    - LSTM units: 64
    - Dropout: 0.3
    - Learning Rate: 1e-4
  - Resulted in a test accuracy of approximately 85.59%.

---

## Inference

- `inference.py` enables running predictions using any of the trained models: Dense, LSTM, or CNN.
- The script accepts the model type and one or more review texts as input.
- Outputs predicted sentiment and probabilities for each input.
- Supports loading saved models and vectorizing input text consistently with training.

---

## Lessons and Next Steps
- Traditional models (Dense, LSTM, CNN) provide a good baseline.
- Hyperparameter tuning improved model performance.
- Transformer models like DistilBERT have shown great potential but experienced local environment issues.
- Future directions:
  - Fine-tuning transformer models on cloud or Kaggle.
  - Exploring ensemble methods.
  - Advanced interpretability and error analysis.

---

## Files
- `main.py`: Model training and evaluation—Dense, LSTM, CNN.
- `tune.py`: Hyperparameter tuning for LSTM.
- `inference.py`: Model inference support for all trained architectures.
- `transformer_finetune.py`: (Planned) Transformer-based fine-tuning with Hugging Face.

---

## Requirements
- Python 3.8+
- TensorFlow 2.x
- Keras Tuner
- Transformers and Datasets libraries from Hugging Face (for transformer experiments)

---

## Notes:
- Environment setup recommended for GPU acceleration.
- Keep libraries updated to minimize compatibility issues.

---

This project demonstrates practical workflows in NLP, model comparison, hyperparameter optimization, and inference deployment, paving the way for exploring transformer-based models.

---