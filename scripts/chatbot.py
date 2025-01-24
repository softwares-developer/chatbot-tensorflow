import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import json

# Baixar o punkt para tokenização
nltk.download('punkt')

# Carregar o modelo treinado
model = load_model('model/chatbot_model.h5')

# Carregar o tokenizer
with open('model/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Carregar o label encoder
with open('model/label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Função para obter resposta
def get_response(text):
    # Tokenizar e padronizar a entrada
    sequence = tokenizer.texts_to_sequences([text])

    # Ajustar maxlen para 2 (tamanho das sequências usadas no treinamento)
    maxlen = 2  # Ajuste conforme o valor usado durante o treinamento
    padded = pad_sequences(sequence, padding='post', maxlen=maxlen)

    # Prever a classe (intenção)
    prediction = model.predict(padded)
    intent_index = np.argmax(prediction)
    intent = label_encoder.inverse_transform([intent_index])[0]

    # Respostas baseadas na intenção
    with open('data/intents.json') as file:
        data = json.load(file)

    # Encontrar a resposta correspondente à intenção
    for intent_obj in data['intents']:
        if intent_obj['intent'] == intent:
            return np.random.choice(intent_obj['responses'])

    return "Desculpe, não entendi."
