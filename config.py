import os

class Config:
    # Base directory for the project
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Directories for data and model storage
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

    # Create directories if they do not exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Hyperparameters for model training
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 10

    # Dataset file format (JSON or CSV)
    DATA_FILE_FORMAT = 'csv'  # 'json' or 'csv' (default is json)

    # Dataset and model paths
    if DATA_FILE_FORMAT == 'json':
        TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.json')
        VAL_DATA_PATH = os.path.join(DATA_DIR, 'val.json')
        TEST_DATA_PATH = os.path.join(DATA_DIR, 'test.json')
    elif DATA_FILE_FORMAT == 'csv':
        TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.csv')
        VAL_DATA_PATH = os.path.join(DATA_DIR, 'val.csv')
        TEST_DATA_PATH = os.path.join(DATA_DIR, 'test.csv')
    else:
        raise ValueError(f"Unsupported data file format: {DATA_FILE_FORMAT}. Please choose 'json' or 'csv'.")

    MODEL_NAME = 'kannada_text_classification_model.h5'

    # Class names for classification (modify according to your dataset)
    CLASSES = ['positive', 'negative']

    # Maximum length of text inputs (for padding)
    MAX_SEQUENCE_LENGTH = 100

    # Tokenizer parameters
    OOV_TOKEN = '<OOV>'
    PADDING_TYPE = 'post'
    TRUNCATING_TYPE = 'post'

    # Random seed for reproducibility
    SEED = 42

    # Directory to store pre-trained embeddings (if any)
    EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'embeddings')
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # Path for saving pre-trained word embeddings (e.g., word2vec, fastText)
    PRETRAINED_EMBEDDINGS = os.path.join(EMBEDDINGS_DIR, 'pretrained_embeddings.vec')

