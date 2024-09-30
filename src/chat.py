import random
import json
import logging
import torch
from model import NeuralNet
from nltk_utils import TextProcessor

# Set up logging
logging.basicConfig(filename='chatbot.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class Chatbot:
    def __init__(self, intents_file='D:/NLP chatbot/data/intents.json', model_file='D:/NLP chatbot/data.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_intents(intents_file)
        self.load_model(model_file)
        self.bot_name = "Lets Chat"

    def load_intents(self, file_path):
        """Load intents from a JSON file."""
        try:
            with open(file_path, 'r') as json_data:
                self.intents = json.load(json_data)
            logging.info(f"Intents loaded from {file_path}.")
        except FileNotFoundError:
            logging.error(f"FileNotFoundError: {file_path} not found.")
            raise
        except json.JSONDecodeError:
            logging.error(f"JSONDecodeError: Could not decode JSON from {file_path}.")
            raise

    def load_model(self, file_path):
        """Load the trained model."""
        try:
            data = torch.load(file_path)
            self.input_size = data["input_size"]
            self.hidden_size = data["hidden_size"]
            self.output_size = data["output_size"]
            self.all_words = data['all_words']
            self.tags = data['tags']
            model_state = data["model_state"]

            self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size).to(self.device)
            self.model.load_state_dict(model_state)
            self.model.eval()
            logging.info(f"Model loaded from {file_path}.")
        except FileNotFoundError:
            logging.error(f"FileNotFoundError: {file_path} not found.")
            raise
        except Exception as e:
            logging.error(f"An error occurred while loading the model: {str(e)}")
            raise

    def chat(self):
        """Start the chat with the user."""
        print("Let's chat! (type 'quit' to exit)")
        while True:
            try:
                sentence = input("You: ")
                if sentence.lower() == "quit":
                    logging.info("User exited the chat.")
                    break

                # Tokenize and prepare the input
                processor = TextProcessor()
                sentence = processor.tokenize(sentence)
                X = processor.bag_of_words(sentence, self.all_words)
                X = X.reshape(1, X.shape[0])
                X = torch.from_numpy(X).to(self.device)

                # Get model prediction
                output = self.model(X)
                _, predicted = torch.max(output, dim=1)

                tag = self.tags[predicted.item()]
                probs = torch.softmax(output, dim=1)
                prob = probs[0][predicted.item()]

                if prob.item() > 0.75:
                    for intent in self.intents['intents']:
                        if tag == intent["tag"]:
                            response = random.choice(intent['responses'])
                            print(f"{self.bot_name}: {response}")
                            logging.info(f"Response: {response} for tag: {tag} with probability: {prob.item()}")
                else:
                    response = "I do not understand..."
                    print(f"{self.bot_name}: {response}")
                    logging.info(f"Response: {response} for tag: {tag} with probability: {prob.item()}")

            except Exception as e:
                logging.error(f"An error occurred: {str(e)}")
                print(f"{self.bot_name}: An error occurred, please try again.")

if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.chat()