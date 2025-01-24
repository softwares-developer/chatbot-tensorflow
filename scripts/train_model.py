from sklearn.preprocessing import LabelEncoder
import numpy as np
import json
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Carregar os dados do intents.json
with open('data/intents.json') as file:
    data = json.load(file)

patterns = []
responses = []
labels = []

# Preparar os dados de entrada
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(intent['responses'])
        labels.append(intent['intent'])

# Tokenização e Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
sequences = tokenizer.texts_to_sequences(patterns)
padded_sequences = pad_sequences(sequences, padding='post', maxlen=5)

# Codificar as labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Salvar o tokenizer e o label_encoder
with open('model/tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('model/label_encoder.pkl', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Criar e treinar o modelo
model = Sequential([
    Dense(128, input_dim=padded_sequences.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(labels_encoded)), activation='softmax')  # Número de classes de saída
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(padded_sequences, labels_encoded, epochs=100, batch_size=8)

# Salvar o modelo treinado
model.save('model/chatbot_model.h5')

print("Modelo treinado e salvo com sucesso!")
