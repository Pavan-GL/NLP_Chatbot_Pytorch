import numpy as np
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import TextProcessor  # Assuming you have this defined
from model import NeuralNet  # Assuming your model class is defined

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ChatbotTrainer:
    def __init__(self, intents_file='D:/NLP chatbot/data/intents.json', num_epochs=1000, batch_size=8, learning_rate=0.001):
        self.intents_file = intents_file
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.all_words = []
        self.tags = []
        self.xy = []
        self.X_train = None
        self.y_train = None

        # Load intents
        self.load_intents()
        self.create_training_data()

        # Hyper-parameters
        #self.input_size = len(self.xy[0][0]) if self.xy else 0
        self.input_size = self.X_train.shape[1] if self.X_train.size > 0 else 0
        self.hidden_size = 8
        self.output_size = len(self.tags)
        logging.info(f'Initialized with input_size: {self.input_size}, output_size: {self.output_size}')

    def load_intents(self):
        """Load intents from a JSON file."""
        try:
            with open(self.intents_file, 'r') as f:
                intents = json.load(f)
            self.process_intents(intents)
        except FileNotFoundError:
            logging.error(f"FileNotFoundError: {self.intents_file} not found.")
            raise
        except json.JSONDecodeError:
            logging.error(f"JSONDecodeError: Could not decode JSON from {self.intents_file}.")
            raise

    def process_intents(self, intents):
        """Process intents to extract patterns and tags."""
        processor = TextProcessor()
        for intent in intents['intents']:
            tag = intent['tag']
            self.tags.append(tag)
            for pattern in intent['patterns']:
                w = processor.tokenize(pattern)
                self.all_words.extend(w)
                self.xy.append((w, tag))

        ignore_words = ['?', '.', '!']
        self.all_words = [processor.stem(w) for w in self.all_words if w not in ignore_words]
        self.all_words = sorted(set(self.all_words))
        self.tags = sorted(set(self.tags))

        logging.info(f"{len(self.xy)} patterns loaded.")
        logging.info(f"{len(self.tags)} tags found: {self.tags}")
        logging.info(f"{len(self.all_words)} unique stemmed words found.")

    def create_training_data(self):
        """Create training data for the model."""
        X_train = []
        y_train = []
        processor = TextProcessor()
        for (pattern_sentence, tag) in self.xy:
            bag = processor.bag_of_words(pattern_sentence, self.all_words)
            X_train.append(bag)
            label = self.tags.index(tag)
            y_train.append(label)

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        print("X_train shape:", self.X_train.shape)
        self.input_size = self.X_train.shape[1] if self.X_train.size > 0 else 0
        print(f'Updated input_size: {self.input_size}')

    def train(self):
        """Train the neural network model."""
        dataset = ChatDataset(self.X_train, self.y_train)
        train_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("Starting training...")
        model = NeuralNet(self.input_size, self.hidden_size, self.output_size).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            for (words, labels) in train_loader:
                words = words.to(device)
                labels = labels.to(dtype=torch.long).to(device)

                # Forward pass
                outputs = model(words)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 100 == 0:
                logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')

        logging.info(f'Final loss: {loss.item():.4f}')
        self.save_model(model)

    def save_model(self, model):
        """Save the trained model and related data."""
        data = {
            "model_state": model.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "all_words": self.all_words,
            "tags": self.tags
        }

        FILE = "data.pth"
        torch.save(data, FILE)
        logging.info(f'Training complete. Model saved to {FILE}')


class ChatDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    try:
        trainer = ChatbotTrainer(num_epochs=1000, batch_size=8, learning_rate=0.001)
        trainer.train()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
