### Chatbot README

This repository contains code for a simple chatbot implemented using Python and TensorFlow. The chatbot is trained on a dataset of intents stored in a JSON file. It utilizes a bag of words approach along with a neural network implemented using TensorFlow to classify user inputs and generate appropriate responses.

#### Requirements
- Python 3.x
- NLTK (Natural Language Toolkit)
- TensorFlow
- tflearn

#### Installation
1. Clone the repository to your local machine.
   ```bash
   git clone https://github.com/Sameed-75210/Chatbot.git
   ```

2. Install the required Python packages using pip.
   ```bash
   pip install -r requirements.txt
   ```

#### Usage
1. Ensure NLTK data is downloaded.
   ```python
   import nltk
   nltk.download('punkt')
   ```

2. Run the `chatbot.py` file to start the chatbot.
   ```bash
   python chatbot.py
   ```

3. Start typing messages to interact with the chatbot. Type "quit" to exit the conversation.

#### Files
- `chatbot.py`: Main Python script containing the chatbot implementation.
- `intents.json`: JSON file containing intents and their corresponding patterns and responses used for training the chatbot.
- `model.tflearn`: Trained model saved after training the chatbot.

#### Dataset
The dataset used to train the chatbot is stored in the `intents.json` file. It contains various intents such as greetings, services, inquiries about the company, and more. Each intent consists of patterns (user inputs) and their corresponding responses.

#### How it Works
1. The chatbot tokenizes user inputs and converts them into a bag of words representation.
2. It utilizes a neural network implemented using TensorFlow to classify the input and predict the appropriate intent.
3. Upon predicting the intent, the chatbot selects a response from the corresponding intent in the dataset and outputs it to the user.

#### Credits
This project was developed by Muhammad Sameed during an internship at SayabiDevs, as part of a learning exercise in natural language processing and chatbot development.

For any inquiries or suggestions, feel free to contact the author at muhammadsameed2002@gmail.com.
