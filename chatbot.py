import nltk
import numpy
import tensorflow
import tflearn 
import random
from nltk.stem.lancaster import LancasterStemmer

nltk.download('punkt')

import json
with open('intents.json') as file:
    data = json.load(file) #loads the contents of the file into a dictionary data.

stemmer = LancasterStemmer() # creates an instance of the LancasterStemmer class.

words = []  # is a list to hold all the words from the patterns in the intents.json file.
labels = [] # is a list to hold the unique tags of the intents.
docs_x = [] # is a list to hold the tokenized words of the patterns.
docs_y = [] # is a list to hold the corresponding tags of the patterns.

for intent in data['intents']:  #loops through each intent in the 'data' dictionary.
    for pattern in intent['patterns']: #loops through each pattern of the current intent.
        wrds = nltk.word_tokenize(pattern)  #tokenizes the words in the pattern using the word_tokenize function from nltk.
        words.extend(wrds) #adds the tokenized words to the words list.
        docs_x.append(wrds) #adds the tokenized words to the docs_x list.
        docs_y.append(intent["tag"]) #adds the tag of the current intent to the docs_y list.
        
    if intent['tag'] not in labels:  #checks if the tag of the current intent is not in the labels list. If so, it adds the tag to the labels list.
        labels.append(intent['tag'])
# print(words)
# Stem the words and remove duplicates, then sort the words
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

# Sort the labels
labels = sorted(labels)

# Lists to hold the training and output data
training = []
output = []

# Empty output list with a size of len(labels)
out_empty = [0 for _ in range(len(labels))]

# Loop through each tokenized words and corresponding tag
for x, doc in enumerate(docs_x):
    bag = []

     # Stem the words in the current tokenized words
    wrds = [stemmer.stem(w.lower()) for w in doc]

    # Check if the word is in the current tokenized words
    # If so, set the value of the corresponding word in the bag list to 1
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)


    # Create a output list with all values set to 0
    output_row = out_empty[:]


    # Set the value of the corresponding tag to 1
    output_row[labels.index(docs_y[x])] = 1

    # Add the bag list to the training list
    training.append(bag)
    # Add the output list to the output list
    output.append(output_row)

# Convert the training and output lists to numpy arrays
training = numpy.array(training)
output = numpy.array(output)

# Reset the default tensorflow graph
tensorflow.compat.v1.reset_default_graph()

# Define the neural network
# Creating the Neural Network with tflearn library
net = tflearn.input_data(shape=[None, len(training[0])])

# Adding fully connected layers to the network
net = tflearn.fully_connected(net, 8) # first hidden layer with 8 neurons

net = tflearn.fully_connected(net, 8) # Second hidden layer with 8 neurons

# output layer with number of neurons equal to number of categories and activation function is softmax
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# Training the model with defined network architecture
model = tflearn.DNN(net)

# training for 1000 epochs with batch size of 8 and showing the metric for each epoch
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)

# saving the trained model for future use
model.save("model.tflearn") 

# Function to generate bag of words for input sentence
def bag_of_words(s, words):
    # Initializing bag of words with 0s
    bag = [0 for _ in range(len(words))]

    # Tokenizing the input sentence
    s_words = nltk.word_tokenize(s)
    # stemming the tokenized words and converting to lowercase
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    # Checking if the tokenized word is present in the words list
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


# Function to chat with the bot
def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Predicting the category for input sentence
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        # Checking if the predicted probability is greater than 0.7
        # if results[results_index] > 0.7:
            # Getting the responses for the category
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

            # Printing random response from the responses list
            print(random.choice(responses))
        # else:
            # print("I didn't understand that, please try again.")

# Starting the chat with the bot
chat()
