import tensorflow as tf
from load_and_preprocess import load_dataset, create_text_vectorizer, preprocess_dataset
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

def build_dense_model(vocab_size, embedding_dim=64, input_length=200):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=input_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_lstm_model(vocab_size, embedding_dim=64, input_length=200):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=input_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_cnn_model(vocab_size, embedding_dim=64, input_length=200):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=input_length),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def plot_history(histories, model_names):
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    for history, name in zip(histories, model_names):
        plt.plot(history.history['accuracy'], label=f'{name} train')
        plt.plot(history.history['val_accuracy'], label=f'{name} val')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    for history, name in zip(histories, model_names):
        plt.plot(history.history['loss'], label=f'{name} train')
        plt.plot(history.history['val_loss'], label=f'{name} val')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

def main():
    # Load and preprocess data
    train_ds_raw, test_ds_raw, info = load_dataset()
    vectorize_layer = create_text_vectorizer(train_ds_raw)

    train_ds = preprocess_dataset(train_ds_raw, vectorize_layer)
    test_ds = preprocess_dataset(test_ds_raw, vectorize_layer, shuffle=False)

    vocab_size = vectorize_layer.vocabulary_size()
    embedding_dim = 64
    max_len = 200

    models = {
        'Dense': build_dense_model(vocab_size, embedding_dim, max_len),
        'LSTM': build_lstm_model(vocab_size, embedding_dim, max_len),
        'CNN': build_cnn_model(vocab_size, embedding_dim, max_len),
    }

    histories = []
    for name, model in models.items():
        print(f'\nTraining {name} model...')
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        checkpoint = ModelCheckpoint(f'best_{name.lower()}_model.h5', monitor='val_loss', save_best_only=True)

        history = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=10,
            callbacks=[early_stop, checkpoint]
        )

        loss, accuracy = model.evaluate(test_ds)
        print(f'{name} model test accuracy: {accuracy:.4f}')
        histories.append(history)

    # Plot comparison graphs
    plot_history(histories, list(models.keys()))

if __name__ == "__main__":
    main()

