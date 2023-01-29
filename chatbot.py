import nltk
import numpy
import tensorflow
import tflearn 
import random
from nltk.stem.lancaster import LancasterStemmer

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

print("Words: ")
print(words)
