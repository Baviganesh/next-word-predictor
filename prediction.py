import numpy as np
import pickle
import heapq
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from os import path

class Prediction:
    def __init__(self) -> None:
        self.WORD_LENGTH = 1
        self.tokenizer = RegexpTokenizer(r'\w+')
        text = open('corpus.txt').read().lower()
        words = self.tokenizer.tokenize(text)
        self.unique_words = np.unique(words)
        self.unique_word_index = dict((c, i) for i, c in enumerate(self.unique_words))
        self.model = self.get_model(words)
        dct = self.generate_freq(words)
        dct1 = self.generate_bi_grams_freq(words)
        self.prob_table = self.create_prob_table(dct,dct1)


    def get_words_prediction(self,line):
        predicted_words = []
        try:
            seq = self.tokenizer.tokenize(line.lower())[-1]
            given_list = self.predict_completions(seq, 3)
            n = self.generate_ngrams(self.tokenizer.tokenize(line.lower())[-1],given_list)
            for i in n:
                k=self.unique_word_index[i[0]]
                m=self.unique_word_index[i[1]]
                next_pred =  {"name" : i[1],"accuracy" : round((self.prob_table[k][m] * 100),2)}
                predicted_words.append(next_pred)
            predicted_words = sorted(predicted_words, key=lambda i: i['accuracy'], reverse=True)
            return predicted_words
        except:
            return []

    def predict_completions(self, text, n=3):
        if text == "":
            return("0")
        x = self.prepare_input(text)
        preds = self.model.predict(x, verbose=0)[0]
        next_indices = self.sample(preds,n)
        return [self.unique_words[idx] for idx in next_indices]
        
    def prepare_input(self, text):
        x = np.zeros((1, self.WORD_LENGTH, len(self.unique_words)))
        for t, word in enumerate(text.split()):
            print(word)
            x[0, t, self.unique_word_index[word]] = 1
        return x

    def sample(self,preds, top_n=3):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return heapq.nlargest(top_n, range(len(preds)), preds.take)

    def generate_ngrams(self,prev_word,predicted_list):
        l = []
        for item in predicted_list : 
            l.append([prev_word,item])
        return l
    
    def find1(self,s,dct1):
        try:
            return dct1[s]
        except:
            return 0

    def generate_bi_grams_freq(self,words):
        l=[]
        i=0
        while(i<len(words)):
            l.append(words[i:i+2])
            i=i+1
        l=l[:-1]
        dct1={}
        for i in l:
            st=" ".join(i)
            dct1[st]=0
        for i in l:
            st=" ".join(i)
            dct1[st]+=1
        return dct1

    def generate_freq(self,words):
        dct={}
        for i in words:
            dct[i]=0
        for i in words:
            dct[i]+=1
        return dct

    def create_prob_table(self,dct,dct1):
        n=len(self.unique_words)
        prob_table=[[]*n for i in range(n)]
        for i in range(n):
            denominator = dct[self.unique_words[i]]
            for j in range(n):
                numerator = self.find1(self.unique_words[i]+" "+self.unique_words[j],dct1)
                prob_table[i].append(float("{:.3f}".format(numerator/denominator)))
        return prob_table

    def get_model(self,words):
        if not path.exists("keras_next_word_model.h5"):
            self.create_model(words)
        return load_model('keras_next_word_model.h5')
    
    def create_model(self,words):
        prev_words = []
        next_words = []
        for i in range(len(words) - self.WORD_LENGTH):
            prev_words.append(words[i:i + self.WORD_LENGTH])
            next_words.append(words[i + self.WORD_LENGTH])
        model = Sequential()
        model.add(LSTM(128, input_shape=(self.WORD_LENGTH, len(self.unique_words))))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.unique_words),activation='softmax'))

        X = np.zeros((len(prev_words), self.WORD_LENGTH, len(self.unique_words)), dtype=bool)
        Y = np.zeros((len(next_words), len(self.unique_words)), dtype=bool)

        for i, each_words in enumerate(prev_words):
            for j, each_word in enumerate(each_words):
                X[i, j, self.unique_word_index[each_word]] = 1
            Y[i, self.unique_word_index[next_words[i]]] = 1

        adam = Adam(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        history = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=25, shuffle=True).history

        model.save('keras_next_word_model.h5')
        pickle.dump(history, open("history.p", "wb"))

        #model = load_model('keras_next_word_model.h5')
        #history = pickle.load(open("history.p", "rb"))
        #return model
