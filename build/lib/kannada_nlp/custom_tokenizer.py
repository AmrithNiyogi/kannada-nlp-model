import re
from .syllable_tokenizer import SyllableTokenizer

class CustomTokenizer:
    def __init__(self):
        """
        Initializes the custom tokenizer by creating an instance of SyllableTokenizer.
        """
        self.syllable_tokenizer = SyllableTokenizer()

    def word_tokenize(self, text):
        """
        Tokenizes text into words using a regular expression.

        Args:
        - text: The input text string.

        Returns:
        - A list of words in the text.
        """
        return re.findall(r"\b\w+\b", text)

    def syllable_tokenize(self, text):
        """
        Tokenizes text into syllables using the syllable tokenizer.

        Args:
        - text: The input text string.

        Returns:
        - A list of syllables in the text.
        """
        return self.syllable_tokenizer.tokenize(text)

    def tokenize(self, text, level="word"):
        """
        Tokenizes text at the specified level (word or syllable).

        Args:
        - text: The input text string.
        - level: The tokenization level, either 'word' or 'syllable'. Defaults to 'word'.

        Returns:
        - A list of tokens at the specified level.

        Raises:
        - ValueError: If the tokenization level is not 'word' or 'syllable'.
        """
        if level == "word":
            return self.word_tokenize(text)
        elif level == "syllable":
            return self.syllable_tokenize(text)
        else:
            raise ValueError("Unsupported tokenization level: {}".format(level))
