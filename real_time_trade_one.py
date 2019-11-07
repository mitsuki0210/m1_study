from datetime import datetime
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import oandapy
import configparser
import pytz
import datetime
from datetime import datetime, timedelta
# Kerasの使用するコンポーネントをインポートしましょう
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM, Embedding
from keras.layers import Dropout
from keras import metrics
from keras.models import model_from_json
import numpy as np
import re
from data_append import make_data
from sklearn.metrics import accuracy_score
from keras import metrics
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from bunrui_func import bunrui_func, pickle_dump, pickle_load
import pickle
from ex import one_hour_predict
from get_price_change import get_oanda_data


csv_data = "BBC_news.csv"


while True:

    # 時間が59分以外の場合は58秒間時間を待機する
    if (datetime.now().minute != 59):
    #59分ではないので1分(58秒)間待機します(誤差がないとは言い切れないので58秒です)
        time.sleep(58)
        print("BBBBBBBBBBBBBBB")
        continue

  
    # 59分になりましたが正確な時間に測定をするために秒間隔で59秒になるまで抜け出せません
    while datetime.now().second != 59:
            # 00秒ではないので1秒待機
            time.sleep(1)
    # 処理が早く終わり二回繰り返してしまうのでここで一秒間待機します
    time.sleep(1)

    data = make_data(csv_data)

    data = data[0]

    level = bunrui_func(data,10)

    if(level == 0　or level == 3):
        print("取引を見送ります。")

    elif(level == 1 or level == 2):
        print("取引を行います。")

    df = get_oanda_data()
    

    

        

  




