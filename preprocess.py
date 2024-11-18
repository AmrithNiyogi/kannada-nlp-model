# In preprocess.py
import csv
import json
import pandas as pd
from config import Config
from kannada_nlp.utils import NLPUtils
from concurrent.futures import ProcessPoolExecutor

def load_csv(input_file):
    """Load CSV data."""
    df = pd.read_csv(input_file)
    return df['text'].tolist(), df['label'].tolist()

def load_dataset(input_file):
    """Load dataset based on the file format (CSV or JSON)."""
    if input_file.endswith('.csv'):
        return load_csv(input_file)
    elif input_file.endswith('.json'):
        with open(input_file, 'r') as f:
            data = json.load(f)
        texts = [entry['text'] for entry in data]
        labels = [entry['label'] for entry in data]
        return texts, labels
    else:
        raise ValueError(f"Unsupported file format: {input_file}. Please provide a CSV or JSON file.")

# Update the preprocess functions to use CSV if required
def preprocess_text_data(input_file, output_file):
    """Preprocess text data and save to a new file."""
    texts, labels = load_dataset(input_file)

    # Preprocess texts (you can modify the preprocessing function in NLPUtils)
    processed_texts = [NLPUtils.preprocess_text(text) for text in texts]

    # Create the output data
    output_data = [{'text': text, 'label': label} for text, label in zip(processed_texts, labels)]

    # Save preprocessed data
    NLPUtils.save_json(output_data, output_file)
    print(f"Preprocessed data saved to {output_file}")

def preprocess_text_data_parallel(input_file, output_file, workers=4):
    """Preprocess text data in parallel."""
    texts, labels = load_dataset(input_file)

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=workers) as executor:
        processed_texts = list(executor.map(NLPUtils.preprocess_text, texts))

    # Create the output data
    output_data = [{'text': text, 'label': label} for text, label in zip(processed_texts, labels)]

    # Save preprocessed data
    NLPUtils.save_json(output_data, output_file)
    print(f"Preprocessed data saved to {output_file}")

