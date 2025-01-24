from flask import Flask, request, jsonify
import sys
import os
from flask_cors import CORS  # Importando o CORS

# Adicionar o diretório 'scripts' ao caminho para garantir que o Flask possa encontrar o chatbot.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

import chatbot  # Agora o Python conseguirá importar o chatbot.py da pasta 'scripts'

# Criando a aplicação Flask e habilitando o CORS
app = Flask(__name__)
CORS(app)  # Permitir CORS para todas as origens

@app.route("/chat", methods=["POST"])
def chat():
    # Obter a entrada do usuário
    user_input = request.json.get("message")

    # Obter a resposta do chatbot
    response = chatbot.get_response(user_input)

    # Retornar a resposta
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
