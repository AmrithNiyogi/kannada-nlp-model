import numpy as np
from tensorflow.keras.models import load_model
from kannada_nlp.custom_tokenizer import CustomTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration
MAX_SEQ_LENGTH = 150
TOKENIZER_LEVEL = "syllable"  # Change to "word" if needed

def preprocess_sample_texts(texts, tokenizer_level="word", token_to_index=None):
    """
    Preprocess sample texts for prediction.

    Args:
    - texts: List of input texts to tokenize.
    - tokenizer_level: Level of tokenization ('word' or 'syllable').
    - token_to_index: Pre-built vocabulary mapping tokens to indices.

    Returns:
    - padded_sequences: Padded tokenized sequences.
    """
    # Initialize tokenizer
    tokenizer = CustomTokenizer()

    # Tokenize the texts
    tokenized_texts = [tokenizer.tokenize(text, level=tokenizer_level) for text in texts]

    # Convert tokens to indices
    tokenized_indices = [[token_to_index.get(token, 0) for token in tokens] for tokens in tokenized_texts]

    # Pad the sequences
    padded_sequences = pad_sequences(tokenized_indices, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")

    return padded_sequences

def load_vocab(vocab_path):
    """
    Load vocabulary from a file.

    Args:
    - vocab_path: Path to the vocabulary file.

    Returns:
    - token_to_index: Dictionary mapping tokens to indices.
    """
    token_to_index = {}
    with open(vocab_path, "r", encoding="utf-8") as vocab_file:
        for idx, token in enumerate(vocab_file):
            token_to_index[token.strip()] = idx + 1
    return token_to_index

def main():
    """
    Main function to load the model and make predictions.
    """
    # Load the trained ensemble model
    model = load_model("trained_ensemble_model.h5")
    print("Model loaded successfully.")

    # Load the vocabulary
    vocab_path = "path/to/vocab.txt"  # Replace with the correct vocabulary file path
    token_to_index = load_vocab(vocab_path)
    print("Vocabulary loaded.")

    # Sample texts for prediction
    sample_texts = [
        "ಕನ್ನಡ ಕವಿತೆಯ ಉತ್ತಮ ಉದಾಹರಣೆ",
        "ಸಾಹಿತ್ಯದಲ್ಲಿ ತತ್ತ್ವಶಾಸ್ತ್ರದ ಪ್ರಾಮುಖ್ಯತೆ"
    ]

    # Preprocess the texts
    preprocessed_texts = preprocess_sample_texts(sample_texts, tokenizer_level=TOKENIZER_LEVEL, token_to_index=token_to_index)

    # Make predictions
    predictions = model.predict([preprocessed_texts, preprocessed_texts])  # Both inputs for CNN and LSTM
    predicted_classes = np.argmax(predictions, axis=1)

    # Print predictions
    print("Predictions:")
    for text, label in zip(sample_texts, predicted_classes):
        print(f"Text: {text} => Predicted Label: {label}")

if __name__ == "__main__":
    main()
