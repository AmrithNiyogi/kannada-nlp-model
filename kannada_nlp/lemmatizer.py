import re


class KannadaLemmatizer:
    def __init__(self):
        # Example simple lemmatization rules for Kannada
        self.suffix_rules = {
            "ಗಳು": "",  # plural suffix
            "ನಿರು": "",  # action-related suffix
            "ಗೆ": "",  # dative suffix
            "ನ": "",  # verb suffix
            "ು": "",  # verb suffix
            "ಎ": "",  # verb suffix
            "ವು": "",  # verb suffix
        }

        self.special_cases = {
            "ಮನುಷ್ಯ": "ಮಾನವ",  # Example: special case for humans (manuṣya -> mānava)
        }

    def lemmatize(self, word):
        """
        Lemmatizes a Kannada word by applying a set of predefined rules
        and handling special cases.
        """
        # Handle special cases
        if word in self.special_cases:
            return self.special_cases[word]

        # Apply suffix removal rules
        for suffix, replacement in self.suffix_rules.items():
            if word.endswith(suffix):
                return word[:-len(suffix)] + replacement

        return word  # If no rule applies, return the word unchanged

    def batch_lemmatize(self, text):
        """
        Lemmatizes a batch of text (sentence or list of words).

        Args:
        - text (str): The input text to lemmatize.

        Returns:
        - str: The lemmatized version of the input text.
        """
        words = text.split()
        lemmatized_words = [self.lemmatize(word) for word in words]
        return " ".join(lemmatized_words)
