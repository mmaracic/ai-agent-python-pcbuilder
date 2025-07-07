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
ORACLE_HOME is a variable sample
```
code ~/.profile
#add variable lines at the bottom of the file:  
     export ORACLE_HOME=/usr/lib/oracle/11.2/client64
```
# Agents


# Start app
* Activate the local python environment .venv
* In the folder where application is, run:
```
uvicorn main:app --app-dir . --host 127.0.0.1 --port 8000 --reload --reload-dir .
```
App-dir option sets the runtime folder of the app to root project folder which results in package names as expected by .py files. 

# App access
Access:  
http://localhost:8000/docs  
To see FastAPI swagger page with information

Available POST API is on:  
http://localhost:8000/setup
http://localhost:8000/query (uses text body)

# Features
* Openrouter
* Short term memory + trimming
* Agent graph
* Tool use

# ToDo
 * Long term memory
 * Streaming support
 * Structured output
 * Human in the loop
 * MCP
 * RAG
 * Multiagentic system

 # Problems
 * Can not even try to parse complex web pages (e.g. stores) on its own, specialised tools might help. 