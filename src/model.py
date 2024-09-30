import torch
import torch.nn as nn
import logging

# Set up logging
logging.basicConfig(filename='model_training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initialize the Neural Network model.

        Parameters:
        - input_size: Number of input features.
        - hidden_size: Number of neurons in the hidden layers.
        - num_classes: Number of output classes.
        """
        super(NeuralNet, self).__init__()

        if input_size <= 0 or hidden_size <= 0 or num_classes <= 0:
            logging.error("Invalid size parameters provided to NeuralNet.")
            raise ValueError("input_size, hidden_size, and num_classes must be positive integers.")

        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

        logging.info(f'NeuralNet initialized with input_size={input_size}, hidden_size={hidden_size}, num_classes={num_classes}')

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x: Input tensor.

        Returns:
        - Output tensor after passing through the network.
        """
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)  # no activation and no softmax at the end
        return out

# Example usage
if __name__ == "__main__":
    try:
        model = NeuralNet(input_size=10, hidden_size=5, num_classes=3)
        logging.info("NeuralNet model instance created successfully.")
    except Exception as e:
        logging.error(f"Failed to create NeuralNet instance: {e}")
