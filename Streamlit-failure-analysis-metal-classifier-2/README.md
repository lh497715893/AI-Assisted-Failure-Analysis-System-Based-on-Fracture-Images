# failure-metal-classifer

CRD project "Development of Artificial Intelligence (AI) assisted Failure Analysis Devices/System Based on Fracture Images" Project user guideline



Prerequisites:
1. MacOS or Windows 7 or above system

2. Python3.7 (https://www.python.org/downloads/release/python-379/)

3. Install pip package (https://pip.pypa.io/en/stable/installing/)

3. Open cmd of windows (or terminal of MacOS)

Create virtual environment: 

#Check the version of Python

python3.7 -V

#Check and install upgraded pip

python3.7 -m pip install pip --upgrade

#Install piping virtual environment

python3.7 -m pip install pipenv

#Direct to your path of your project

cd PATH-TO-YOUR-PROJECT/

#Check the version of the pipenv

pipenv

#Create virtual environment for the project

pipenv shell

Install necessary python packages for the interface and display the classification result of the metal fracture types:

pip3 install -r requirements.txt

Run the project by streamlit comment line:

streamlit run rps_app.py
