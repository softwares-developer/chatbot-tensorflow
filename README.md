# Chatbot com TensorFlow

Este projeto é um chatbot simples utilizando o TensorFlow para processamento de linguagem natural (NLP). O modelo é treinado com dados de intenções (perguntas e respostas) e pode ser integrado facilmente a um site ou aplicativo.

## Estrutura do Projeto

- **data/**: Contém os dados de treinamento (intents.json).
- **model/**: Contém o modelo treinado (chatbot_model.h5) e os objetos necessários para o processamento (tokenizer, label_encoder).
- **scripts/**: Scripts de pré-processamento, treinamento do modelo e interação com o chatbot.
- **app/**: API Flask para interagir com o chatbot.
- **requirements.txt**: Dependências do projeto.

## Como Usar

1. Instale as dependências:
