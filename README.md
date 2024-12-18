# Kannada NLP Project

A Python package for Natural Language Processing (NLP) tasks in Kannada. This package includes several models, tokenizers, augmenters, and utilities for various NLP applications like sentiment analysis, named entity recognition (NER), text classification, and more, specifically designed for the Kannada language.

---

## Features

- **Text Classification**: Classify Kannada text into predefined categories using models like BiLSTM, CNN, and ensemble approaches.
- **Sentiment Analysis**: Analyze the sentiment of Kannada text (positive, negative, neutral).
- **Named Entity Recognition (NER)**: Extract named entities from Kannada text.
- **Text Preprocessing**: Tokenization, lemmatization, stemming, and augmentation of Kannada text.
- **Custom Tokenizers**: Specialized tokenizers for syllables and words in Kannada.
- **Visualization**: Visualize model performance, data distributions, and more.
- **Augmentation**: Augment Kannada text data to increase model robustness.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmrithNiyogi/kannada-nlp-model.git
cd kannada-nlp-model
```

### 2. Create and Activate Virtual Environment (Optional but Recommended)
For Python 3.x:
```bash
python3 -m venv venv
source venv/bin/activate   # For Linux/macOS
.\venv\Scripts\activate    # For Windows
```

### 3. Install Dependencies
Run the following command to install all the necessary dependencies:

```bash
pip install -r requirements.txt
```

### 4. Install the Package Locally
Once dependencies are installed, you can install the package locally using the following command:

```bash
pip install .
```

---


## Project Structure

```arduino
kannada_nlp_project/
├── kannada_nlp/
│   ├── __init__.py
│   ├── attention_layer.py
│   ├── bilstm_model.py
│   ├── classifier.py
│   ├── cnn_model.py
│   ├── custom_tokenizer.py
│   ├── early_stopping.py
│   ├── ensemble_model.py
│   ├── hierarchical_classifier.py
│   ├── kannada_augmentor.py
│   ├── kannada_embedding_layer.py
│   ├── lemmatizer.py
│   ├── metrics.py
│   ├── ner.py
│   ├── sentiment_analysis.py
│   ├── syllable_tokenizer.py
│   ├── utils.py
│   ├── visualization.py
│   ├── config.py
│   ├── train.py
│   └── preprocess.py
├── setup.py
├── README.md
├── requirements.txt
└── LICENSE
```

### Main Files:
- **train.py**: Contains model training scripts and evaluation routines.
- **preprocess.py**: Includes data preprocessing functions (tokenization, lemmatization, etc.).
- **sentiment_analysis.py**: Functions for analyzing the sentiment of Kannada text.
- **ner.py**: Contains functions for Named Entity Recognition on Kannada text.
- **visualization.py**: Functions for generating plots and graphs (e.g., confusion matrices).
- **utils.py**: General utility functions used across different modules.
- **config.py**: Configuration file where hyperparameters and paths are defined.

---

## Contributions
We welcome contributions to improve the Kannada NLP Project. To contribute:

- Fork the repository. 
- Create a new branch.
- Implement your changes.
- Submit a pull request with a clear description of the changes.

---

## License
This project is licensed. See the LICENSE file for details.

---

## Acknowledgements
- TensorFlow and Keras for deep learning models.
- scikit-learn for machine learning utilities.
- NLTK and other libraries for text processing.
- All contributors and open-source libraries that helped build this project.