from kannada_nlp.custom_tokenizer import CustomTokenizer
from kannada_nlp.ner import KannadaNER
import numpy as np

# Mock setup
VOCAB = {"ಮಹೇಂದ್ರ": 1, "ಸಿಂಗ್": 2, "ಧೋನಿ": 3, "ಭಾರತದ": 4, "ಕ್ರಿಕೆಟ್": 5, "ಆಟಗಾರ": 6}
TAGS = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC"}
VOCAB_SIZE = len(VOCAB) + 1
EMBEDDING_DIM = 128
NUM_TAGS = len(TAGS)
MAX_SEQ_LENGTH = 10

# Initialize the KannadaNER model
ner_model = KannadaNER(VOCAB_SIZE, EMBEDDING_DIM, NUM_TAGS, MAX_SEQ_LENGTH)

# Load pre-trained weights (mock example)
# ner_model.model.load_weights("path/to/saved_model.h5")  # Uncomment if weights are available

# Tokenizer
tokenizer = CustomTokenizer()

# Input sentence
sentence = "ಮಹೇಂದ್ರ ಸಿಂಗ್ ಧೋನಿ ಭಾರತದ ಕ್ರಿಕೆಟ್ ಆಟಗಾರ"

# Tokenization
tokens = tokenizer.tokenize(sentence, level="word")
print(f"Tokens: {tokens}")

# Convert tokens to indices
token_indices = [VOCAB.get(token, 0) for token in tokens]  # Unknown tokens mapped to 0
print(f"Token Indices: {token_indices}")

# Pad token indices to match MAX_SEQ_LENGTH
token_indices_padded = np.pad(token_indices, (0, MAX_SEQ_LENGTH - len(token_indices)), mode="constant")
token_indices_padded = np.expand_dims(token_indices_padded, axis=0)  # Batch dimension

# Mock model prediction
def viterbi_decode(logits, sequence_length):
    """
    Perform Viterbi decoding to find the most likely sequence of tags.

    Args:
    - logits: A 2D numpy array of shape (sequence_length, num_tags) containing scores for each tag.
    - sequence_length: A list containing the length of the sequence.

    Returns:
    - path_score: The score of the best path.
    - best_path: A list of tag indices representing the most likely sequence of tags.
    """
    sequence_length = sequence_length[0]  # Assuming batch size of 1 for simplicity
    num_tags = logits.shape[1]

    # Initialize variables
    viterbi_scores = np.zeros((sequence_length, num_tags))
    backpointers = np.zeros((sequence_length, num_tags), dtype=int)

    # Initialization step (start with the scores of the first token)
    viterbi_scores[0] = logits[0]

    # Recursion step (compute the best scores and backpointers)
    for t in range(1, sequence_length):
        for curr_tag in range(num_tags):
            # Compute the score for transitioning to `curr_tag`
            transition_scores = viterbi_scores[t - 1] + logits[t, curr_tag]
            best_prev_tag = np.argmax(transition_scores)
            viterbi_scores[t, curr_tag] = transition_scores[best_prev_tag]
            backpointers[t, curr_tag] = best_prev_tag

    # Termination step (find the best final tag)
    best_last_tag = np.argmax(viterbi_scores[-1])
    path_score = viterbi_scores[-1, best_last_tag]

    # Traceback step (reconstruct the best path)
    best_path = [best_last_tag]
    for t in range(sequence_length - 1, 0, -1):
        best_last_tag = backpointers[t, best_last_tag]
        best_path.insert(0, best_last_tag)

    return path_score, best_path


def mock_prediction(input_sequence):
    """
    Simulate model predictions for demonstration purposes.
    """
    # Mock logits output
    mock_logits = np.random.rand(input_sequence.shape[1], NUM_TAGS)
    sequence_length = [input_sequence.shape[1]]
    _, tags = viterbi_decode(mock_logits, sequence_length)
    return tags

# Predict NER tags
predicted_tags = mock_prediction(token_indices_padded)

# Map indices back to tags
tagged_output = [(token, TAGS.get(tag, "O")) for token, tag in zip(tokens, predicted_tags)]
print("\nNER Output:")
for token, tag in tagged_output:
    print(f"{token} -> {tag}")
