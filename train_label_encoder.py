from sklearn.preprocessing import LabelEncoder
import pickle
import json

# Carregar as intenções do arquivo intents.json
with open('data/intents.json') as file:
    data = json.load(file)

# Crie uma lista de todas as intenções (como "greeting", "goodbye", etc.)
intents = [intent['intent'] for intent in data['intents']]

# Inicialize e treine o LabelEncoder com todas as intenções
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(intents)

# Salvar o label_encoder em um arquivo .pkl
with open('model/label_encoder.pkl', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Salvar as intenções codificadas em um arquivo .npy (opcional para diagnóstico)
import numpy as np
np.save('model/labels_encoded.npy', labels_encoded)

print("Label Encoder e Labels salvos com sucesso!")
