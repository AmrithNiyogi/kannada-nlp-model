import json
import re


class NLPUtils:

    @staticmethod
    def preprocess_text(text):
        """Basic preprocessing: remove non-alphanumeric characters except Kannada characters and spaces."""
        # Removing English punctuation but keeping Kannada unicode characters and spaces
        text = re.sub(r'[^\u0C80-\u0CFF\s]', '', text)  # Keep Kannada unicode characters and spaces
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
