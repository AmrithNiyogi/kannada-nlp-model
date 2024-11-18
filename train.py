# kannada_nlp/train.py

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from config import Config
from kannada_nlp.utils import NLPUtils

def load_data():
    """Load and preprocess the dataset."""
    # Load training data
    texts, labels = NLPUtils.load_dataset(Config.TRAIN_DATA_PATH)

    # Split into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=Config.SEED)

    return x_train, x_val, y_train, y_val

def tokenize_data(x_train, x_val):
    """Tokenize the text data."""
    tokenizer = Tokenizer(oov_token=Config.OOV_TOKEN)
    tokenizer.fit_on_texts(x_train)

    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_val_seq = tokenizer.texts_to_sequences(x_val)

    x_train_padded = pad_sequences(x_train_seq, maxlen=Config.MAX_SEQUENCE_LENGTH, padding=Config.PADDING_TYPE,
                                   truncating=Config.TRUNCATING_TYPE)
    x_val_padded = pad_sequences(x_val_seq, maxlen=Config.MAX_SEQUENCE_LENGTH, padding=Config.PADDING_TYPE,
                                 truncating=Config.TRUNCATING_TYPE)

    return x_train_padded, x_val_padded, tokenizer

def build_model(vocab_size):
    """Build and compile the LSTM model."""
    model = Sequential([
        Embedding(vocab_size, 128, input_length=Config.MAX_SEQUENCE_LENGTH),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dense(64, activation='relu'),
        Dense(len(Config.CLASSES), activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=Config.LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    """Train the model."""
    x_train, x_val, y_train, y_val = load_data()
    x_train_padded, x_val_padded, tokenizer = tokenize_data(x_train, x_val)

    model = build_model(len(tokenizer.word_index) + 1)

    history = model.fit(x_train_padded, y_train, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE,
                        validation_data=(x_val_padded, y_val))

    model.save(os.path.join(Config.MODEL_DIR, Config.MODEL_NAME))
    print("Model trained and saved.")

if __name__ == '__main__':
    train_model()
