import random

class KannadaAugmentor:
    def __init__(self):
        """
        Initializes the KannadaAugmentor with a dictionary of common Kannada words and their synonyms.
        """
        self.synonym_dict = {
            "ನಮಸ್ಕಾರ": ["ನಮಸ್ಕಾರಗಳು", "ವಂದನೆಗಳು", "ಸಲಾಮು"],
            "ಭಾಷೆ": ["ಭಾಷಾವಾರು", "ಭಾಷಾಭಿಮುಖ", "ಅಕ್ಷರಬರಹ"],
            "ಅಧ್ಯಯನ": ["ಶಿಕ್ಷಣ", "ಕಲಿಕೆ", "ಅಭ್ಯಾಸ"],
            "ಮಾತು": ["ಶಬ್ದ", "ಉಪನ್ಯಾಸ", "ಸಂಭಾಷಣೆ"],
            "ಸಿರಿ": ["ಸಂಪತ್ತು", "ಸೌಂದರ್ಯ", "ಅರಸು"],
            "ಹೃದಯ": ["ಮನ", "ಅಂತರಂಗ", "ಮನೋಭಾವ"],
        }

    def synonym_replacement(self, text):
        """
        Replaces words in the text with their synonyms based on a predefined dictionary.

        Args:
        - text: The input text to augment.

        Returns:
        - augmented_text: The text with synonym replacements.
        """
        words = text.split()
        augmented_text = [
            random.choice(self.synonym_dict[word]) if word in self.synonym_dict else word
            for word in words
        ]
        return " ".join(augmented_text)

    def random_insertion(self, text, insert_word="ನಮಸ್ಕಾರ"):
        """
        Randomly inserts a word into the input text at a random position.

        Args:
        - text: The input text to augment.
        - insert_word: The word to insert randomly into the text (default: "ನಮಸ್ಕಾರ").

        Returns:
        - augmented_text: The text with the randomly inserted word.
        """
        words = text.split()
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, insert_word)
        return " ".join(words)

    def random_deletion(self, text, p=0.2):
        """
        Randomly deletes words from the text with a given probability.

        Args:
        - text: The input text to augment.
        - p: The probability of deleting each word (default: 0.2).

        Returns:
        - augmented_text: The text with randomly deleted words.
        """
        words = text.split()
        if len(words) == 1:
            return text
        return " ".join([word for word in words if random.random() > p])

    def augment_text(self, text, techniques=None):
        """
        Applies multiple augmentation techniques to the input text.

        Args:
        - text: The input text to augment.
        - techniques: A list of augmentation techniques to apply (optional). If None, applies all techniques.

        Returns:
        - augmented_text: The augmented text after applying the selected techniques.
        """
        if techniques is None:
            techniques = ["synonym_replacement", "random_insertion", "random_deletion"]

        augmented_text = text
        for technique in techniques:
            if hasattr(self, technique):
                augmented_text = getattr(self, technique)(augmented_text)
        return augmented_text
