import re

class SyllableTokenizer:
    def __init__(self):
        # Define Kannada vowels, consonants, and diacritics
        self.vowels = "ಅಆಇಈಉಊಋಎಏಐಒಓಔಅಂಅಃ"
        self.consonants = "ಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲವಶಷಸಹಳ"
        self.diacritics = "್"  # Halant or Virama, which is used to combine consonants

        # Adjusted pattern to correctly handle vowel and consonant combinations
        self.syllable_pattern = r"([ಕ-ಹ]{1}[ಾ-ೆ-ೊ-್]?)|([ಅಆಇಈಉಊಋಎಏಐಒಓಔಅಂಅಃ])"  # Fixed vowel range

        # Extended pattern to handle conjunct consonants (like 'ಕ್ಷ', 'ತ್ರ', etc.)
        self.conjunct_pattern = r"(ಕ್ಷ|ತ್ರ|ಕ್ಷ|ಕ್ಷ|ಮಥ|ಬ್ರ|ಶ್ರ|ಹ್ಮ|ಕ್ಞ|ಕ್ಲ|ಫ್ರ|ದ್ರ|ಅಸ್ತ|ಸ್ಥ|ಪ್ರ)"

    def tokenize(self, text):
        """Tokenizes Kannada text into syllables."""
        # Normalize the text to handle issues like extra spaces
        text = self.normalize_text(text)

        syllable_list = []

        # First, we apply the conjunct pattern to handle conjunct consonants
        text = re.sub(self.conjunct_pattern, lambda x: x.group(0), text)

        # Then, apply the syllable pattern to capture consonant-vowel combinations or just vowels
        syllable_list.extend(re.findall(self.syllable_pattern, text))

        # Return the resulting list of syllables
        # Flatten the list, as re.findall() returns tuples (either consonant + vowel or vowel alone)
        syllable_list = [item[0] if item[0] else item[1] for item in syllable_list]

        return syllable_list

    def normalize_text(self, text):
        """Normalize the input text (optional)."""
        # Remove excessive spaces and normalize text (optional)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()