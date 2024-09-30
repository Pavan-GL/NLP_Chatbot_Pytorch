import numpy as np
import nltk
import logging
from nltk.stem.porter import PorterStemmer

# Initialize logging
logging.basicConfig(filename='text_processing.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('punkt')

class TextProcessor:
    def __init__(self):
        """Initialize the TextProcessor with a PorterStemmer."""
        self.stemmer = PorterStemmer()
        logging.info("TextProcessor initialized.")

    def tokenize(self, sentence):
        """
        Split sentence into array of words/tokens.
        A token can be a word, punctuation character, or number.
        """
        try:
            tokens = nltk.word_tokenize(sentence)
            logging.info(f"Tokenized sentence: {tokens}")
            return tokens
        except Exception as e:
            logging.error(f"Error tokenizing sentence: {sentence}. Error: {str(e)}")
            raise

    def stem(self, word):
        """
        Find the root form of the word.
        """
        try:
            stemmed_word = self.stemmer.stem(word.lower())
            logging.info(f"Stemmed word: {word} -> {stemmed_word}")
            return stemmed_word
        except Exception as e:
            logging.error(f"Error stemming word: {word}. Error: {str(e)}")
            raise

    def bag_of_words(self, tokenized_sentence, words):
        """
        Return bag of words array:
        1 for each known word that exists in the sentence, 0 otherwise.
        """
        try:
            # Stem each word
            sentence_words = [self.stem(word) for word in tokenized_sentence]
            # Initialize bag with 0 for each word
            bag = np.zeros(len(words), dtype=np.float32)

            for idx, w in enumerate(words):
                if w in sentence_words:
                    bag[idx] = 1
            
            logging.info(f"Bag of words for sentence: {tokenized_sentence} -> {bag}")
            return bag
        except Exception as e:
            logging.error(f"Error creating bag of words for sentence: {tokenized_sentence}. Error: {str(e)}")
            raise

# Example usage:
if __name__ == "__main__":
    nltk.download('punkt')
    
    processor = TextProcessor()
    sentence = "Hello, how are you?"
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]

    tokens = processor.tokenize(sentence)
    bag = processor.bag_of_words(tokens, words)

    print(f"Tokens: {tokens}")
    print(f"Bag of Words: {bag}")
