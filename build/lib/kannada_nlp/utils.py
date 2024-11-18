# kannada_nlp/utils.py

import json
import re
from nltk.tokenize import word_tokenize

class NLPUtils:

    @staticmethod
    def preprocess_text(text):
        """Basic preprocessing: remove non-alphanumeric characters."""
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text

    @staticmethod
    def load_dataset(file_path):
        """Load a dataset from a JSON file."""
        with open(file_path, 'r') as file:
            data = json.load(file)
        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]
        return texts, labels

    @staticmethod
    def save_json(data, file_path):
        """Save data to a JSON file."""
        with open(file_path, 'w') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
