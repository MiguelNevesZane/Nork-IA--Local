"""
main.py — ponto de entrada do projeto.
Lança o Motor IA Avançado (Ollama + deepseek-r1:8b).

Para rodar:
    python main.py

Pré-requisitos:
    1. Ollama instalado e rodando em outro terminal: ollama serve
    2. Modelo baixado: ollama pull deepseek-r1:8b
    3. pip install requests   (já instalado)
"""

from motor_ia_avancado import chat

if __name__ == "__main__":
    chat()
