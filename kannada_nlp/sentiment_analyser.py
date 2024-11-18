import numpy as np
from kannada_nlp import CustomTokenizer, KannadaAugmentor

class SentimentAnalyser:
    def __init__(self, model, tokenizer_vocab, max_seq_length):
        self.model = model
        self.tokenizer = CustomTokenizer()
        self.tokenizer.vocab = tokenizer_vocab
        self.max_seq_length = max_seq_length
        self.augmentor = KannadaAugmentor()

    def preprocess_text(self, text):
        # Augment and tokenize text
        augmented_text = self.augmentor.augment_text(text)
        tokenized_text = self.tokenizer.encode(augmented_text)
        padded_sequence = self.tokenizer.pad_sequences([tokenized_text], self.max_seq_length)
        return np.array(padded_sequence)

    def predict_sentiment(self, text):
        preprocessed_text = self.preprocess_text(text)
        prediction = self.model.predict(preprocessed_text)
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        return sentiment, float(prediction)
