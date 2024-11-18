from kannada_nlp.custom_tokenizer import CustomTokenizer

if __name__=='__main__':
    # Sample Kannada text for tokenization
    text_kannada = "ಈ ಒಂದು ಉದಾಹರಣೆಯು ಕನ್ನಡು ಭಾಷೆಯಲ್ಲಿಯೇ ಟೋಕನೈಝೇಷನ್ ತೋರಿಸುತ್ತದೆ."

    # Instantiate the CustomTokenizer
    custom_tokenizer = CustomTokenizer()

    # Word Tokenization
    word_tokens = custom_tokenizer.tokenize(text_kannada, level="word")
    print("Word Tokens (Kannada):")
    print(word_tokens)

    # Syllable Tokenization
    syllable_tokens = custom_tokenizer.tokenize(text_kannada, level="syllable")
    print("\nSyllable Tokens (Kannada):")
    print(syllable_tokens)

    # Invalid Tokenization Level (This will raise an exception)
    try:
        invalid_tokens = custom_tokenizer.tokenize(text_kannada, level="sentence")
    except ValueError as e:
        print("\nError:", e)
