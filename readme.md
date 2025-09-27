# Python 3 local environment
* Install Python3
* If python3 is not recognised as python command
```
sudo apt install python-is-python3
```
* Install pip (will not work if pip was installed using apt-get):
```
sudo apt install python3-pip
```
In MSYS:
```
pacman -S python3-pip
```
* Install possibility to create virtual environments
```
sudo apt install python3-venv
```
* Add install folder to ~/.profile

For Windows
* Install Python 3
* Install pip:
```
python -m ensurepip --upgrade
```
```
python -m pip install --upgrade pip
```

* Add python and pip path <python_folder>\Scripts (e.g. C:\Python38\Scripts) to Windows path 

## PIP
The custom pip libraries need to be installed in virtual environment (otherwise for environment maintained for apt-get we will get - error: externally-managed-environment)

Check if virtual environment is active (if its not active it will write /usr/bin/python, if it is cmd will be prefixed by name of virtual env and this will print env path)
```
which python (Linux)
```
```
where python (Windows)
```
Create virtual environment in .venv subfolder
```
python3 -m venv .venv
```
Activate environment in .venv subfolder:
```
source .venv/bin/activate (Linux)
```
```
.\.venv\bin\activate (Windows)
```
In the virtual environment it will now be possible to install any needed libraries using:
```
python3 -m pip install requests
```
```
python3 -m pip install -r requirements.txt
```

To deactivate the current virtual environment use:
```  
deactivate
```
# Set environment variables
Project configuration is provided via a `.env` file in the repository root (a distributable template is `.env.sample`).

Steps:
1. Copy the sample file:
```
cp .env.sample .env
```
2. Edit `.env` and add your real keys (never commit secrets).
3. Always run the app (or any command needing the variables) through the virtual environment and `python -m dotenv run -- <command>` so the variables are loaded automatically.

Example (same as the VS Code task `start-app`):
```bash
source .venv/bin/activate
python -m dotenv run -- uvicorn main:app --app-dir . --host 127.0.0.1 --port 8000 --reload --reload-dir .
```

UI (same pattern as `start-ui` task):
```bash
source .venv/bin/activate
python -m dotenv run -- streamlit run ui.py
```

Quick test that an env var is visible (example using GOOGLE_API_KEY):
```bash
python -m dotenv run -- python -c "import os; print(os.getenv('GOOGLE_API_KEY'))"
```

This avoids manual exporting and keeps secrets scoped only to the invoked process.
# Start app
Option A: VS Code tasks (preferred)
* Backend API: Task `start-app`
* UI: Task `start-ui`

Option B: Run manually in a terminal (full commands)

Backend (FastAPI + Uvicorn):
```bash
source .venv/bin/activate
python -m pip install -r requirements.txt  # if not already installed
python -m dotenv run -- uvicorn main:app --app-dir . --host 127.0.0.1 --port 8000 --reload --reload-dir .
```

UI (Streamlit):
```bash
source .venv/bin/activate
python -m pip install -r requirements.txt  # if not already installed
python -m dotenv run -- streamlit run ui.py
```

Notes:
* `python -m dotenv run -- <cmd>` injects variables from `.env` only for that command.
* `--app-dir .` pins the project root so imports resolve as expected.
* `--reload` / `--reload-dir .` enable hot-reload during development.
# App access
Access:  
http://localhost:8000/docs  
To see FastAPI swagger page with information

Available POST API is on:  
http://localhost:8000/setup
http://localhost:8000/query (uses text body)

# Testing
To run tests run:
```
pytest
```
To get coverage with pytest-cov package installed use:
```
pytest-cov
```

# Features
* Openrouter
* Short term memory + trimming
* Agent graph
* Tool use
* Structured output (item_extractor_agent.py)
* Multiagentic system (agents are in main.py and item_extractor_agent.py)

# ToDo
 * Long term memory
 * Streaming support
 * Human in the loop
 * MCP
 * RAG
 * Evaluations
 
 # Problems
 