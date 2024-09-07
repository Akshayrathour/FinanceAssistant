from tkinter import *
from tkcalendar import DateEntry
import datetime as dt

import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
# from  keras.models import load_model

import os
import json
import random
import pickle

from typing import Union

import nltk
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import  Optimizer

class BasicAssistant:

    def __init__(self, intents_data: Union[str, os.PathLike, dict], method_mappings: dict = {}, hidden_layers: list = None, model_name: str = "basic_model") -> None:

        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)

        if isinstance(intents_data, dict):
            self.intents_data = intents_data
        else:
            if os.path.exists(intents_data):
                with open(intents_data, "r") as f:
                    self.intents_data = json.load(f)
            else:
                raise FileNotFoundError

        self.method_mappings = method_mappings
        self.model = None
        self.hidden_layers = hidden_layers
        self.model_name = model_name
        self.history = None

        self.lemmatizer = nltk.stem.WordNetLemmatizer()

        self.words = []
        self.intents = []

        self.training_data = []

    def _prepare_intents_data(self, ignore_letters: tuple = ("!", "?", ",", ".")):
        documents = []

        for intent in self.intents_data["intents"]:
            if intent["tag"] not in self.intents:
                self.intents.append(intent["tag"])

            for pattern in intent["patterns"]:
                pattern_words = nltk.word_tokenize(pattern)
                self.words += pattern_words
                documents.append((pattern_words, intent["tag"]))

        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_letters]
        self.words = sorted(set(self.words))

        empty_output = [0] * len(self.intents)

        for document in documents:
            bag_of_words = []
            pattern_words = document[0]
            pattern_words = [self.lemmatizer.lemmatize(w.lower()) for w in pattern_words]
            for word in self.words:
                bag_of_words.append(1 if word in pattern_words else 0)

            output_row = empty_output.copy()
            output_row[self.intents.index(document[1])] = 1
            self.training_data.append([bag_of_words, output_row])

        random.shuffle(self.training_data)
        self.training_data = np.array(self.training_data, dtype="object")

        X = np.array([data[0] for data in self.training_data])
        y = np.array([data[1] for data in self.training_data])

        return X, y

    def fit_model(self, optimizer: Optimizer = None, epochs: int = 200):
        X, y = self._prepare_intents_data()

        if self.hidden_layers is None:
            self.model = Sequential()
            self.model.add(InputLayer(input_shape=(None, X.shape[1])))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(y.shape[1], activation='softmax'))
        else:
            self.model = Sequential()
            self.model.add(InputLayer(input_shape=(None, X.shape[1])))
            for layer in self.hidden_layers:
                self.model.add(layer)
            self.model.add(Dense(y.shape[1], activation='softmax'))


        # if optimizer is None:
        #     optimizer = Adam(learning_rate=0.01)

        self.model.compile('adam',loss="categorical_crossentropy", metrics=["accuracy"])

        self.history = self.model.fit(X, y, epochs=epochs, batch_size=5, verbose=1)

    def save_model(self):
        self.model.save(f"{self.model_name}.h5", self.history)
        pickle.dump(self.words, open(f'{self.model_name}_words.pkl', 'wb'))
        pickle.dump(self.intents, open(f'{self.model_name}_intents.pkl', 'wb'))
    
    def load_model(self):
        self.model = load_model(f'{self.model_name}.h5')
        self.words = pickle.load(open(f'{self.model_name}_words.pkl', 'rb'))
        self.intents = pickle.load(open(f'{self.model_name}_intents.pkl', 'rb'))

    def _predict_intent(self, input_text: str):
        input_words = nltk.word_tokenize(input_text)
        input_words = [self.lemmatizer.lemmatize(w.lower()) for w in input_words]

        input_bag_of_words = [0] * len(self.words)

        for input_word in input_words:
            for i, word in enumerate(self.words):
                if input_word == word:
                    input_bag_of_words[i] = 1

        input_bag_of_words = np.array([input_bag_of_words])

        predictions = self.model.predict(input_bag_of_words, verbose=0)[0]
        predicted_intent = self.intents[np.argmax(predictions)]

        max_prob = np.max(predictions)
        # print(max_prob)
        # if max_prob < self.confidence_threshold:
        #     return None
        # predicted_intent = self.intents[np.argmax(predictions)]

        return predicted_intent

    def process_input(self, input_text: str):
        predicted_intent = self._predict_intent(input_text)

        try:
            for intent in self.intents_data["intents"]:
                if intent["tag"] == predicted_intent:
                    l=[random.choice(intent["responses"]),predicted_intent]
                    return l
            
            if predicted_intent in self.method_mappings:
                self.method_mappings[predicted_intent]()
        except IndexError:
            return "I don't understand. Please try again."

chat=BasicAssistant('intent_logs.json')
chat.load_model()

my_portfolio={'aapl':10,'msft':10,'tsla':20}

def show_stocks():
    for i in my_portfolio:
        insert_msg(f"\n{i}:{str(my_portfolio[i])}")

def buy_stocks():
    ticker=updt_enter.get("1.0",END)
    c=count_enter.get("1.0",END)
    for i in my_portfolio:
        if i==ticker:
            a=max(int(c),my_portfolio[i])
            my_portfolio[i]+=a


def sell_stocks():
    ticker=updt_enter.get("1.0",END)
    c=count_enter.get("1.0",END)
    for i in my_portfolio:
        if i==ticker:
            a=max(int(c),my_portfolio[i])
            my_portfolio[i]-=a

def worth_stocks():
    sum=0
    c=0
    prev=0
    start=dt.datetime(2023,12,1)
    end=dt.datetime(2024,1,2)
    for i in my_portfolio:
        data=yf.download(i,start,end)
        data=data.iloc[:,-1:]
        data=data.to_numpy()
        prev=data[-2:-1]
        sum+=data[-1:]
        c+=1
    insert_msg(f"\nBot: Total Worth= {str(sum)}")
    insert_msg(f"\nTotal gains= {str((sum-prev)/c)}")

def advice_stocks():
    start=dt.datetime(2023,12,1)
    end=dt.datetime(2024,1,1)
    for i in my_portfolio:
        df=yf.download(i,start,end)
        df=df.iloc[:,-3:]
        df=df.to_numpy()
        df=df[-3:]
        model = load_model('stmodel.h5')
        res=model.predict(df)
        price=(res[0]+res[1]+res[2])/3
        a=int(df[-1][0])
        if(price>a):
            insert_msg(f"\nBot: Uptrend detected for {i}, can sell")
        elif(price<=a):
            insert_msg(f"\nBot: Downtrend detected for {i}, do not sell")


mappings={'stocks':show_stocks,'worth':worth_stocks,'advice':advice_stocks}

def predictor():
    start=dt.datetime(2023,12,1)
    end=dt.datetime(2024,1,1)
    ticker=predictor_enter.get()
    df=yf.download(ticker,start,end)
    df=df.iloc[:,-3:]
    df=df.to_numpy()
    df=df[-3:]
    model = load_model('stmodel.h5')
    res=model.predict(df)
    price=(res[0]+res[1]+res[2])/3
    predictor_cls_lbl.configure(text=str(price))


def insert_msg(msg):
    if not msg:
        return
    inter_screen.configure(state=NORMAL)
    inter_screen.insert(END,msg)
    inter_screen.configure(state=DISABLED)

def msg_entered(event):
    msg=user_entry.get("1.0",END)
    ans=chat.process_input(msg)
    func=ans[1]
    insert_msg(f"\nYou: {msg}")
    insert_msg(f"\nBot: {ans[0]}")
    user_entry.delete("1.0",END)
    if func in mappings:
        mappings[func]()

def visualize():
    from_date = cal_from.get_date()
    to_date = cal_to.get_date()

    start = dt.datetime(from_date.year, from_date.month, from_date.day)
    end = dt.datetime(to_date.year, to_date.month, to_date.day)

    # Load Ticker From Entry And Download Data
    ticker = visual_enter.get()
    data = yf.download(ticker, start, end)

    # Restructure Data Into OHLC Format
    data = data[['Open', 'High', 'Low', 'Close']]

    # Reset Index And Convert Dates Into Numerical Format
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].map(mdates.date2num)


    # Adjust Style Of The Plot
    ax = plt.subplot()
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_title('{} Share Price'.format(ticker), color='white')
    # ax.figure.canvas.set_window_title('NeuralNine Stock Visualizer v0.1 Alpha')
    ax.set_facecolor('black')
    ax.figure.set_facecolor('#121212')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis_date()

    # Plot The Candlestick Chart
    candlestick_ohlc(ax, data.values, width=0.5, colorup='#00ff00')
    plt.show()

root=Tk()
root.title('Finance_assisstant')
root.configure(width=700,height=400,background="#2c2c2c")
root.resizable(height=False,width=False)

inter_screen=Text(root,height=17,width=50)
inter_screen.configure(cursor="arrow",state=DISABLED)
inter_screen.place(x=10,y=10)
scroll_bar=Scrollbar(inter_screen)
scroll_bar.place(relheight=1,relx=0.974)
scroll_bar.configure(command=inter_screen.yview)

user_entry=Text(root,height=5,width=50)
user_entry.place(x=10,y=300)
user_entry.focus()
user_entry.bind("<Return>",msg_entered)

predictor_labl=Label(root,bg="White",text="Stock Price predictor",width=37)
predictor_labl.place(x=425,y=10)
txt_lbl=Label(root,text="Enter the ticker:",bg="#2c2c2c",fg="white")
txt_lbl.place(x=430,y=35)

predictor_enter=Entry(root,width=30)
predictor_enter.place(x=430,y=60)

predict_btn=Button(root,text="predict",command=predictor)
predict_btn.place(x=630,y=60)

txt2_lbl=Label(root,text="Predicted price:",bg="#2c2c2c",fg="white")
txt2_lbl.place(x=430,y=80)
predictor_cls_lbl=Label(root,bg="White",width=37,height=1)
predictor_cls_lbl.place(x=425,y=100)

stock_visual_label=Label(root,bg="white",text="Stock Visualizer",width=37)
stock_visual_label.place(x=425,y=140)
txt3_lbl=Label(root,text="Enter the ticker:",bg="#2c2c2c",fg="white")
txt3_lbl.place(x=430,y=170)

visual_enter=Entry(root,width=30)
visual_enter.place(x=430,y=190)

predict_btn=Button(root,text="visualize",command=visualize)
predict_btn.place(x=630,y=190)

label_from = Label(root, text="From:",bg="#2c2c2c",fg="white")
label_from.place(x=430,y=220)
cal_from = DateEntry(root, width=27, year=2023, month=1, day=1)
cal_from.place(x=430, y=240)

label_to = Label(root, text="To:",bg="#2c2c2c",fg="white")
label_to.place(x=430,y=260)
cal_to = DateEntry(root, width=27)
cal_to.place(x=430,y=280)

txt4_lbl=Label(root,text="Enter the ticker for updation:",bg="#2c2c2c",fg="white")
txt4_lbl.place(x=430,y=300)

updt_enter=Entry(root,width=30)
count_enter=Entry(root,width=20)
sell_btn=Button(root,text='sell',command=sell_stocks)
buy_btn=Button(root,text='buy',command=buy_stocks)
updt_enter.place(x=430,y=320)
count_enter.place(x=430,y=360)
sell_btn.place(x=630,y=320)
buy_btn.place(x=630,y=360)

root.mainloop()
