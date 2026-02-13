import tensorflow as tf
import tensorflow_datasets as tfds

def load_dataset():
    # Loading the IMBD dataset using the TensorFlow datasets
    # Loads train and test splits with supervised labels
    dataset, info = tfds.load('imdb_reviews', with_info = True, as_supervised = True)

    return dataset['train'], dataset['test'], info


# Creating a text vectorizer (tokenizer) layer for converting text to integer sequences
def create_text_vectorizer(train_ds, max_features = 10000, max_len = 200):
    # The layer converts words to integer indices and pads outputs to max_len
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens = max_features,
        output_mode = 'int',
        output_sequence_length = max_len
    )

    # Extracting only text (not labels) for adaptation (building vocabulary)
    train_text = train_ds.map(lambda text, label: text)
    # building the vocabulary from the training dataset
    vectorize_layer.adapt(train_text)
    return vectorize_layer

# Preprocess the dataset with tokenization and batching
def preprocess_dataset(dataset, vectorize_layer, batch_size = 32, shuffle = True):
    # Internal function to process one (text, label) pair at a time
    def vectorize_text(text, label):
        return vectorize_layer(text), label

    if shuffle:
        dataset = dataset.shuffle(10000)

    # Map text to vectorized form, batch it, and prefetch for performance
    dataset = dataset.map(vectorize_text).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def main():
    train_ds, test_ds, info = load_dataset()
    
    # Show info
    print(info)

    # Take one example from the trainind dataset and print it
    for example, label in train_ds.take(1):
        print('Example review: ', example.numpy())
        print('Label: ', label.numpy())

    vectorize_layer = create_text_vectorizer(train_ds)

    train_data = preprocess_dataset(train_ds, vectorize_layer)
    test_data = preprocess_dataset(test_ds, vectorize_layer, shuffle = False)

    # Inspect a sample batch for verification of shapes and labels
    for texts, labels in train_data.take(1):
        print('Batch text tensor shape: ', texts.shape)    # should be (batch_size, max_len)
        print('Batch label tensor: ', labels.numpy())      # batch of labels

if __name__ == "__main__":
    main()