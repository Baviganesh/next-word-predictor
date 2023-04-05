# next-word-predictor
Predicting three words that comes next with Tensorflow. Implement LSTM to develop the model.

## Project Intro
The purpose of this project is to train next word predicting model. Model should be able to suggest three words as the next word after user has input word. The model is trained with technology dataset. Python as backend and JavaScript/HTML as Frontend.

### Methods Used
* Language Prediction
* Natural Language Processing
* LSTM

### Technologies
* Python
* Python Flask
* Tensorflow, Keras
* Js, HTML

## Project Description
* `corpus.txt` - file to train model
* `predictor.py` - python file that contains the algorithm
* `app.py` - python file that contains the flask code to talk to predictor.py
* `WordCount.xlsx` - Excel file to record the words and counts

## Process Flow
- frontend development
- data collection
- data processing/cleaning
- words tokenizing
- model training

## Getting Started

### Prerequisites/ Steps to run the application

Python Version Used : 3.10.7

Steps to Deploy and Run the Project:
1. Create Virtual Environment
	- For Windows
		
		python -m venv .venv
	- For mac/linux
		
		python3 -m venv .venv
	
2. Select Virtual Environment
	- For Windows
	
		.venv\Scripts\activate
		
	- For mac/linux
	
		source .venv/bin/activate
		
	
3. Upgrade pip
	- For Windows
		
		python -m pip install --upgrade pip
	- For mac/linux
		
		python3 -m pip install --upgrade pip
	
4. Install Dependencies
	- pip install flask
	- pip install flask_cors
	- pip install flask_restful
	- pip install pandas
	- pip install nltk
	- pip install keras
	- pip install tensorflow
	- pip install openpyxl
	
5. Run Flask App
	
	python main.py
	
6. Open APP
	
	Enter http://127.0.0.1:5000 or http://localhost:5000 in any browser
	
## Contributing Members

|Name     |
|---------|
| Sangeetha Nair |
| Viviana Lopes |
| Megha Lal |
| Bavithra Ganesan |
