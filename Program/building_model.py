import pandas as pd
from keras.src.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from kannada_nlp.custom_tokenizer import CustomTokenizer
from tensorflow.keras.preprocessing.sequences import pad_sequences
from kannada_nlp.cnn_model import CNNModel
from kannada_nlp.bilstm_model import BiLSTMModel
from kannada_nlp.ensemble_model import EnsembleModel

# Configuration
VOCAB_SIZE = 20000
EMBEDDING_DIM = 128
MAX_SEQ_LENGTH = 150
BATCH_SIZE = 32
EPOCHS = 5

def save_vocab(token_to_index, vocab_file_path):
    """
    Save the vocabulary mapping to a text file.

    Args:
    - token_to_index: Dictionary mapping tokens to indices.
    - vocab_file_path: Path where the vocab file will be saved.
    """
    with open(vocab_file_path, "w", encoding="utf-8") as f:
        for token, idx in token_to_index.items():
            f.write(f"{token}\n")  # Each token on a new line
    print(f"Vocabulary saved to {vocab_file_path}")

def preprocess_data(csv_path, tokenizer_level="word", vocab_file_path="vocab.txt"):
    """
    Load and preprocess the CSV data using CustomTokenizer.

    Args:
    - csv_path: Path to the CSV file.
    - tokenizer_level: Level of tokenization ('word' or 'syllable').
    - vocab_file_path: Path where the vocab file will be saved.

    Returns:
    - x_train: Tokenized and padded training sequences.
    - x_val: Tokenized and padded validation sequences.
    - y_train: Encoded training labels.
    - y_val: Encoded validation labels.
    - vocab_size: Vocabulary size based on the tokenizer.
    - num_classes: Number of unique classes for classification.
    """
    # Load the CSV file
    data = pd.read_csv(csv_path, names=["Category", "Content"], header=None)

    # Separate features and labels
    x_data = data["Content"]
    y_data = data["Category"]

    # Encode labels to integers
    label_encoder = LabelEncoder()
    y_data_encoded = label_encoder.fit_transform(y_data)
    num_classes = len(label_encoder.classes_)

    # Split into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data_encoded, test_size=0.2, random_state=42)

    # Initialize custom tokenizer
    tokenizer = CustomTokenizer()

    # Tokenize data
    x_train_tokens = [tokenizer.tokenize(text, level=tokenizer_level) for text in x_train]
    x_val_tokens = [tokenizer.tokenize(text, level=tokenizer_level) for text in x_val]

    # Build vocabulary
    vocab = {token for tokens in x_train_tokens for token in tokens}
    token_to_index = {token: idx + 1 for idx, token in enumerate(vocab)}
    vocab_size = len(token_to_index) + 1

    # Save vocabulary to file
    save_vocab(token_to_index, vocab_file_path)

    # Convert tokens to indices
    x_train_indices = [[token_to_index[token] for token in tokens if token in token_to_index] for tokens in x_train_tokens]
    x_val_indices = [[token_to_index[token] for token in tokens if token in token_to_index] for tokens in x_val_tokens]

    # Pad sequences
    x_train_padded = pad_sequences(x_train_indices, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")
    x_val_padded = pad_sequences(x_val_indices, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")

    return x_train_padded, x_val_padded, y_train, y_val, vocab_size, num_classes

def main(train_csv, tokenizer_level="word"):
    """
    Main function to train the ensemble model with custom tokenizer.

    Args:
    - train_csv: Path to the training CSV file.
    - tokenizer_level: Level of tokenization ('word' or 'syllable').
    """
    # Preprocess data
    x_train, x_val, y_train, y_val, vocab_size, num_classes = preprocess_data(train_csv, tokenizer_level)

    # Build CNN and BiLSTM models
    cnn_model_instance = CNNModel(vocab_size, EMBEDDING_DIM, MAX_SEQ_LENGTH)
    cnn_model = cnn_model_instance.build_model()

    bilstm_model_instance = BiLSTMModel(vocab_size, EMBEDDING_DIM, MAX_SEQ_LENGTH)
    bilstm_model = bilstm_model_instance.build_model()

    # Build ensemble model
    ensemble_model_instance = EnsembleModel(lstm_model=bilstm_model, cnn_model=cnn_model)
    ensemble_model = ensemble_model_instance.build_model()

    # Compile the ensemble model
    ensemble_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train the model
    ensemble_model.fit(
        [x_train, x_train],  # Inputs for both CNN and BiLSTM
        y_train,
        validation_data=([x_val, x_val], y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    # Save the trained model
    ensemble_model.save("trained_ensemble_model.h5")
    print("Model training complete and saved as trained_ensemble_model.h5")

if __name__ == "__main__":
    train_csv_path = "data/kn-train.csv"  # Replace with your CSV path
    main(train_csv_path, tokenizer_level="syllable")  # Change to "word" for word-level tokenization