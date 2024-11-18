# kannada_nlp/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

class Visualizer:
    def __init__(self):
        sns.set(style="whitegrid")

    def plot_word_cloud(self, text_data):
        """Generate a word cloud from the text data."""
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text_data))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud')
        plt.show()

    def plot_sentiment_distribution(self, labels):
        """Plot the distribution of sentiment labels."""
        plt.figure(figsize=(8, 5))
        sns.countplot(x=labels)
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, classes):
        """Plot the confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()

    def plot_accuracy_loss(self, history):
        """Plot accuracy and loss curves."""
        # Assuming history is a dictionary with 'accuracy', 'val_accuracy', 'loss', 'val_loss'
        plt.figure(figsize=(12, 5))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_top_n_words_per_sentiment(self, text_data, labels, n=10):
        """Plot the top N words for each sentiment class."""
        df = pd.DataFrame({'text': text_data, 'label': labels})
        top_words = {}

        for sentiment in df['label'].unique():
            words = ' '.join(df[df['label'] == sentiment]['text'])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
            top_words[sentiment] = wordcloud

        # Plotting
        plt.figure(figsize=(15, 10))
        for i, (sentiment, wc) in enumerate(top_words.items()):
            plt.subplot(2, 2, i + 1)
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Top Words for Sentiment: {sentiment}')
        plt.tight_layout()
        plt.show()