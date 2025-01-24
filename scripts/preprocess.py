import nltk
import json
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Baixar o punkt para tokenização
nltk.download('punkt')

# Carregar os dados
with open('data/intents.json') as file:
    data = json.load(file)

patterns = []
responses = []
labels = []
intents = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(intent['responses'])
        labels.append(intent['intent'])
    if intent['intent'] not in intents:
        intents.append(intent['intent'])

# Tokenização
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
sequences = tokenizer.texts_to_sequences(patterns)
padded_sequences = pad_sequences(sequences, padding='post')

# LabelEncoder para categorias (intents)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Salvar objetos necessários para o treinamento e uso posterior
np.save('model/padded_sequences.npy', padded_sequences)
np.save('model/labels_encoded.npy', labels_encoded)

# Salvar tokenizer usando pickle
with open('model/tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Salvar label_encoder
with open('model/label_encoder.pkl', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Dados e modelos salvos com sucesso!")
