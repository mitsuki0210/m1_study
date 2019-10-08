import pandas as pd
import oandapy
import datetime
import numpy as np
from datetime import datetime, timedelta
import pytz
# API接続設定のファイルを読み込む
import configparser
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout


account_id = "1401673"
api_key = "88cb1d105fa6e388042747e4bfea5943-a4d0a01845fd3924f3aaf6ab10fce462"
 
 
# APIへ接続
oanda = oandapy.API(environment="practice", access_token=api_key)
 
# ドル円の現在のレートを取得
res = oanda.get_prices(instruments="GBP_JPY")
 
# 中身を確認
#print(res)

def iso_to_jp(iso):
    date = None
    try:
        date = datetime.strptime(iso, '%Y-%m-%dT%H:%M:%S.%fZ')
        date = pytz.utc.localize(date).astimezone(pytz.timezone("Asia/Tokyo"))
    except ValueError:
        try:
            date = datetime.strptime(iso, '%Y-%m-%dT%H:%M:%S.%f%z')
            date = date.astimezone(pytz.timezone("Asia/Tokyo"))
        except ValueError:
            pass
    return date
 
# datetime -> 表示用文字列
def date_to_str(date):
    if date is None:
        return ''
    return date.strftime('%Y/%m/%d %H:%M:%S')
 
# ドル円の現在のレートを取得
res = oanda.get_prices(instruments="GBP_JPY")
 
#print(date_to_str(iso_to_jp(res['prices'][0]['time'])))


 
for i in range(3):
    if i == 0:
        res_hist = oanda.get_history(instrument="GBP_JPY", granularity="M5", count=5000)
    else:
        res_hist = oanda.get_history(instrument="GBP_JPY", granularity="M5",end=endtime, count=5000)
    res = res_hist.get("candles")
    endtime = res[0]['time']
    if i == 0 : res1 = res
    else :
        for j in range(len(res1)):
            res.append(res1[j])
        res1 = res
    print('res ok', i+1, 'and', 'time =', res1[0]['time'])
 
#データフレームに変換しておく
res2= pd.DataFrame(res1)
 
#取得件数を数えて出力
#print(len(df))
res2["time"] = res2["time"].apply(lambda x: iso_to_jp(x))

print(res2)

df = res2[['time', 'openAsk']]
df.columns = ['time', 'open']

print(df[14799:14800])
#print(df[4799:4800])

window_len = 10

split_date = '2019/05/21 21:25:00'
train, test = df[df['time'] < split_date], df[df['time']>=split_date]
latest = test[:window_len]
del train['time']
del test['time']
del latest['time']
length = len(test)- window_len


def data_maker(data):
  data_lstm_in=[]
  if len(data)==window_len:
    temp = data[:window_len].copy()
    temp = temp / temp.iloc[0] - 1
    data_lstm_in.append(temp)
  for i in range(len(data) - window_len):
      temp = data[i:(i + window_len)].copy()
      temp = temp / temp.iloc[0] - 1
      data_lstm_in.append(temp)
  return data_lstm_in


train_lstm_in = data_maker(train)
lstm_train_out = (train['open'][window_len:].values / train['open'][:-window_len].values)-1
test_lstm_in = data_maker(test)
lstm_test_out = (test['open'][window_len:].values / test['open'][:-window_len].values)-1
latest_lstm_in = data_maker(latest)


def pd_to_np(data_lstm_in):
  data_lstm_in = [np.array(data_lstm_input) for data_lstm_input in data_lstm_in]  #array のリスト
  data_lstm_in = np.array(data_lstm_in) #np.array
  return data_lstm_in
train_lstm_in = pd_to_np(train_lstm_in)
test_lstm_in = pd_to_np(test_lstm_in)
latest_lstm_in = pd_to_np(latest_lstm_in)


def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()
 
    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
 
    model.compile(loss=loss, optimizer=optimizer)
    return model

    np.random.seed(202)
 
# 初期モデルの構築
yen_model = build_model(train_lstm_in, output_size=1, neurons = 20)
 
# データを流してフィッティングさせましょう
yen_history = yen_model.fit(train_lstm_in, lstm_train_out, 
                            epochs=1, batch_size=1, verbose=2, shuffle=True)

empty = []
future_array = np.array(empty)
for i in range(length + 10):
  pred = (((np.transpose(yen_model.predict(latest_lstm_in))+1) * latest['open'].values[0])[0])[0]
  future_array= np.append(future_array,pred)
  data ={'open':[pred]}
  df1 = pd.DataFrame(data)
  latest =pd.concat([latest,df1],axis=0)
  latest.index = range(0,window_len+1)
  latest = latest.drop(0,axis=0)
  latest_lstm_in =pd_to_np(latest_lstm_in)


"""
plt.figure(figsize=(10,8))
plt.plot(df[df['time']< split_date]['time'][window_len:],
         train['open'][window_len:], label='Actual', color='blue')

plt.plot(df[df['time']< split_date]['time'][window_len:],
         ((np.transpose(yen_model.predict(train_lstm_in))+1) * train['open'].values[:-window_len])[0], 
         label='Predicted', color='red')
plt.show()


"""
plt.figure(figsize=(10,8))
plt.plot(df[df['time']>= split_date]['time'][window_len:],
         test['open'][window_len:], label='Actual', color='blue')
plt.plot(df[df['time']>= split_date]['time'][window_len:],
        future_array,label='future',color='green')
plt.show()
