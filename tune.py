import tensorflow as tf
from kerastuner.tuners import RandomSearch
from load_and_preprocess import load_dataset, create_text_vectorizer, preprocess_dataset

def build_lstm_hypermodel(hp):
    # Use fixed vocab size from your dataset info, or a fixed number
    vocab_size = 20000  
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=hp.Int('embedding_dim', min_value=32, max_value=128, step=32),
        input_length=200  # fixed sequence length
    ))
    model.add(tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            hp.Int('lstm_units', min_value=32, max_value=128, step=32)
        )
    ))
    model.add(tf.keras.layers.Dropout(
        hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    ))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    # Load and preprocess dataset
    train_ds_raw, test_ds_raw, _ = load_dataset()
    vectorize_layer = create_text_vectorizer(train_ds_raw)
    
    train_ds = preprocess_dataset(train_ds_raw, vectorize_layer)
    test_ds = preprocess_dataset(test_ds_raw, vectorize_layer, shuffle=False)

    # Setup tuner
    tuner = RandomSearch(
        build_lstm_hypermodel,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='imdb_sentiment_tuning'
    )

    tuner.search(train_ds, validation_data=test_ds, epochs=5)

    # Get the optimal hyperparameters and model
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters:")
    print(f"Embedding dim: {best_hp.get('embedding_dim')}")
    print(f"LSTM units: {best_hp.get('lstm_units')}")
    print(f"Dropout rate: {best_hp.get('dropout_rate')}")
    print(f"Learning rate: {best_hp.get('learning_rate')}")

    best_model = tuner.get_best_models(num_models=1)[0]
    loss, acc = best_model.evaluate(test_ds)
    print(f"Best model test accuracy: {acc:.4f}")

    best_model.save('best_lstm_tuned_model.keras')
    print("Best tuned model saved as 'best_lstm_tuned_model.keras'")

if __name__ == "__main__":
    main()
