import tensorflow as tf
import sys
from load_and_preprocess import create_text_vectorizer, load_dataset

def load_model(model_type='dense'):
    """
    Load the trained model for the specified type.
    """
    model_paths = {
        'dense': 'best_dense_model.h5',
        'lstm': 'best_lstm_model.h5',
        'cnn':  'best_cnn_model.h5'
    }
    path = model_paths.get(model_type, 'best_dense_model.h5')
    model = tf.keras.models.load_model(path)
    return model

def preprocess_text(texts, vectorize_layer):
    texts = tf.constant(texts)
    vectors = vectorize_layer(texts)
    return vectors

def predict_sentiment(model, vectorize_layer, texts):
    vectors = preprocess_text(texts, vectorize_layer)
    probs = model.predict(vectors)
    preds = (probs > 0.5).astype(int)
    return probs.flatten(), preds.flatten()

def main():
    # Parse arguments: first argument is model type, rest are texts
    args = sys.argv[1:]
    model_type = args[0] if len(args) > 1 else 'dense'
    input_texts = args[1:] if len(args) > 1 else [
        "This movie was fantastic! I really enjoyed it.",
        "The plot was boring and predictable."
    ]

    # Load vectorizer and model
    train_ds_raw, _, _ = load_dataset()
    vectorize_layer = create_text_vectorizer(train_ds_raw)
    model = load_model(model_type)

    probs, preds = predict_sentiment(model, vectorize_layer, input_texts)
    for text, prob, pred in zip(input_texts, probs, preds):
        sentiment = 'Positive' if pred == 1 else 'Negative'
        print(f"Review: {text}")
        print(f"Predicted sentiment: {sentiment} (probability: {prob:.3f})\n")

if __name__ == "__main__":
    main()
