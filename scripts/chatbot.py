from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
import json
import numpy as np

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

    # Use o mesmo maxlen que foi utilizado durante o treinamento (por exemplo, 5)
    maxlen = 5  # Ajuste de acordo com o valor usado no treinamento

    # Padronize a sequência
    padded = pad_sequences(sequence, padding='post', maxlen=maxlen)

    # Prever a classe (intenção)
    prediction = model.predict(padded)
    intent_index = np.argmax(prediction)

    try:
        # Verifica se a intenção prevista está no LabelEncoder
        intent = label_encoder.inverse_transform([intent_index])[0]
    except ValueError:
        # Se o índice não for reconhecido, retorna uma mensagem de erro
        intent = "Desculpe, não entendi."

    # Respostas baseadas na intenção
    with open('data/intents.json') as file:
        data = json.load(file)

    # Encontrar a resposta correspondente à intenção
    for intent_obj in data['intents']:
        if intent_obj['intent'] == intent:
            return np.random.choice(intent_obj['responses'])

    return "Desculpe, não entendi."
