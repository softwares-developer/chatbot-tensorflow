import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Carregar e preparar os dados
# Certifique-se de que você tenha pré-processado os dados antes (tokenizer, padded_sequences, labels_encoded)
# Isso inclui a tokenização e o padding das sequências, além da codificação das labels.

# Exemplo de dados fictícios de entrada (substitua por seus dados reais):
padded_sequences = np.load('model/padded_sequences.npy')  # Exemplo de dados pré-processados
labels_encoded = np.load('model/labels_encoded.npy')  # Exemplo de labels codificados

# Definir o modelo
model = Sequential([
    Dense(128, input_dim=padded_sequences.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(labels_encoded)), activation='softmax')  # Número de classes de saída
])

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Resumo do modelo para visualizar a arquitetura
model.summary()

# Treinar o modelo
model.fit(padded_sequences, labels_encoded, epochs=100, batch_size=8)

# Salvar o modelo treinado
model.save('model/chatbot_model.h5')

print("Modelo treinado e salvo com sucesso!")
