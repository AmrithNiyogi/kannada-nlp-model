import re

class KannadaLemmatizer:
    def __init__(self):
        # Example simple lemmatization rules for Kannada
        self.suffix_rules = {
            "ಗಳು": "",    # plural suffix
            "ನಿರು": "",    # action-related suffix
            "ಗೆ": "",      # dative suffix
            "ನ": "",       # verb suffix
            "ು": "",       # verb suffix
            "ಎ": "",       # verb suffix
            "ವು": "",      # verb suffix
            "ವಾದರೆ": "",  # conditional verb form
            "ತೇನೆ": "ಹೋಗು", # remove tense suffixes for future tense
            "ುತ್ತಾನೆ": "ಹೋಗು", # 3rd person singular verb ending
            "ುತ್ತಿದ್ದಾರೆ": "ಹೋಗು", # present continuous verb ending
            "ಲಿರು": "",    # past tense of action verbs
        }

        self.special_cases = {
            "ಮನುಷ್ಯ": "ಮಾನವ",  # Special case for 'human'
            "ಅವನ": "ಅವನು",      # Special case for 'he'
            "ಹೊಳಪುಗೊಳಿಸುವ": "ಹೊಳಪುಗೊಳಿಸು",  # Special case for verb
            "ಗುರುವಿನ": "ಗುರು",  # Special case for possessive
            "ಕೂಡು": "ಕೂಡ",     # Correct form of verb
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
                # Remove suffix and apply replacement if needed
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