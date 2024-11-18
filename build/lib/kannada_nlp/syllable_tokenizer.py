import re


class SyllableTokenizer:
    def __init__(self):
        # Define a regex pattern for Kannada syllables
        self.vowels = "ಅಆಇಈಉಊಋಎಏಐಒಓಔಅಂಅಃ"
        self.consonants = "ಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲವಶಷಸಹಳ"
        self.syllable_pattern = f"[{self.consonants}][{self.vowels}]*"

        # Alternative pattern for handling Kannada syllables with optional signs
        self.syllable_pattern_alternative = re.compile(r'([ಕ-ಹ][್]?[ಾ-ೆ-ೊ-್]?)')

    def tokenize(self, text, use_alternative_pattern=False):
        """Tokenizes Kannada text into syllables."""
        # Normalize the text to remove extra spaces
        text = self.normalize_text(text)

        # Split the text into words
        words = text.split()

        syllable_list = []
        for word in words:
            if use_alternative_pattern:
                syllables = self.extract_syllables_using_alternate_pattern(word)
            else:
                syllables = self.extract_syllables(word)
            syllable_list.extend(syllables)

        return syllable_list

    def extract_syllables(self, word):
        """Extract syllables from a single word using the main syllable pattern."""
        return re.findall(self.syllable_pattern, word)

    def extract_syllables_using_alternate_pattern(self, word):
        """Extract syllables from a single word using the alternative pattern."""
        return self.syllable_pattern_alternative.findall(word)

    def normalize_text(self, text):
        """Normalize the input text (optional)."""
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        return text.strip()