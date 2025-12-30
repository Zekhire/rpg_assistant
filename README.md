# rpg_assistant
First attempt to create RAG algorithm to support RPG GMs.


## Prerequisites
- Windows 11
- Docker Desktop
- NVIDIA GPU
- Python 3.10.11


## Quick Start
1. Adjust **settings.py**
2. In cmd:

```sh
# Ollama setup
docker compose up -d
docker exec -it ollama ollama run llama3 --verbose
docker exec -it ollama ollama run mxbai-embed-large --verbose

# RAG setup
python -m venv venv
venv//Scripts//activate
python -m pip install -r requirements.txt
python ingest.py
python app.py
```

## Note
Made with the use of ChatGPT (https://chatgpt.com/)