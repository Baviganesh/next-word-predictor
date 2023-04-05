from flask import Flask, request, render_template
from flask_cors import CORS
from flask_restful import reqparse
import pandas as pd

from prediction import Prediction

app = Flask(__name__)
cors = CORS(app)

prediction = Prediction()

data_put_args = reqparse.RequestParser()
data_put_args.add_argument("word", type=str, help="Word Typed")
data_put_args.add_argument("isPredict", type=str, help="Is Word Prrdicted (Y/N)")

data_get_args = reqparse.RequestParser()
data_get_args.add_argument("data", type=str, help="Word Typed")

@app.route("/")
def index():
    return render_template('main.html')

@app.get("/prediction")
def get_next_words():
    data = request.args['data']
    return prediction.get_words_prediction(data)

@app.get("/excelData")
def get_excelData():
    data = pd.read_excel("WordCount.xlsx")
    return data.to_json()

@app.put("/excelData")
def put_excelData():
    args = data_put_args.parse_args()
    word = args['word'].lower()
    print("Entered word :",word)
    isPredict = args['isPredict']
    word = format_word(word)
    print("Formatted word : '"+word+"'")
    data = pd.read_excel("WordCount.xlsx")
    if word:
        try:
            counts = data['COUNT'].values
            words = data['WORD'].values
            if(word in words):
                index = data[data['WORD']==word].index.values
                count = counts[index[0]] + 1
                data['COUNT'][index[0]] = count
                data['FLAG'][index[0]] = isPredict
            else:
                new_row = pd.Series({'WORD': word, 'COUNT': 1, 'FLAG': isPredict})
                data = append_row(data, new_row)
            data.to_excel("WordCount.xlsx", index=False)
        except:
            print("Close the Excel File.")
    else:
        print("Word is Blank")
    return data.to_json()

def format_word(word):
    if not word.isalnum():
        sample_list=[]
        for i in word:
            if i.isalnum():
                sample_list.append(i)
        word="".join(sample_list)
    return word.strip()

def append_row(df, row):
    return pd.concat([df, pd.DataFrame([row], columns=row.index)]).reset_index(drop=True)

if __name__ == "__main__":
    app.run()