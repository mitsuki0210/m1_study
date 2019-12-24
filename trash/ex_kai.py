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

# 日付の調整する
# 文字列 -> datetime
def iso_jp(iso):
    date = None
    try:
        date = datetime.strptime(iso, '%Y-%m-%dT%H:%M:%S.%fZ')
        date = pytz.utc.localize(date).astimezone(pytz.timezone("Asia/Tokyo"))
    except ValueError:
        try:
            date = datetime.strptime(iso, '%Y-%m-%dT%H:%M:%S.%f%z')
            date = dt.astimezone(pytz.timezone("Asia/Tokyo"))
        except ValueError:
            pass
    return date
 
# datetime -> 表示用文字列
def date_string(date):
    if date is None:
        return ''
    return date.strftime('%Y/%m/%d %H:%M:%S')


ACCOUNT_ID = "1401673"
ACCESS_TOKEN = "88cb1d105fa6e388042747e4bfea5943-a4d0a01845fd3924f3aaf6ab10fce462"
 
oanda = oandapy.API(environment="practice", access_token=ACCESS_TOKEN)
 

#response = oanda.get_history(instrument="GBP_JPY", granularity="M5")

#res = pd.DataFrame(response["candles"])
 
# 最初の5行を表示させる
#res.head()
 
#res['time'] = res['time'].apply(lambda x: iso_jp(x))
#res['time'] = res['time'].apply(lambda x: date_string(x))
 
#print(res.head())
 
#df = res[['time', 'openAsk', 'closeAsk', 'highAsk', 'lowAsk', 'volume']]
#df.columns = ['time', 'open', 'close', 'high', 'low', 'volume']
 

for i in range(6):
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
res2["time"] = res2["time"].apply(lambda x: iso_jp(x))
res2['time'] = res2['time'].apply(lambda x: date_string(x))


#print(res2)

df = res2[['time','closeAsk']]
df.columns = ['time','close']





#print(df[479:480])
print(df[24999:25000])


split_date = '2019/11/18 11:00:00'
train, test = df[df['time'] < split_date], df[df['time']>=split_date]
del train['time']
del test['time']



# windowを設定
window_len = 12

batch_size = 1

# LSTMへの入力用に処理（訓練）
train_lstm_in = []
for i in range(len(train) - window_len):
    temp = train[i:(i + window_len)].copy()
    for col in train:      
        temp.loc[:, col] = temp[col] / temp[col].iloc[0] - 1
    train_lstm_in.append(temp)
 
lstm_train_out = (train['close'][window_len:].values / train['close'][:-window_len].values)-1

"""
lstm_train_out_bi = []
for j in range(len(lstm_train_out)):
    if(lstm_train_out[j] >= 0):
        lstm_train_out_bi.append(1)
    else:
        lstm_train_out_bi.append(0)
"""


# LSTMへの入力用に処理（テスト）
test_lstm_in = []
for i in range(len(test) - window_len):
    temp = test[i:(i + window_len)].copy()
    for col in test:
        temp.loc[:, col] = temp[col] / temp[col].iloc[0] - 1
    test_lstm_in.append(temp)
lstm_test_out = (test['close'][window_len:].values / test['close'][:-window_len].values)-1
 



# PandasのデータフレームからNumpy配列へ変換しましょう
train_lstm_in = [np.array(train_lstm_input) for train_lstm_input in train_lstm_in]

train_lstm_in = np.array(train_lstm_in)
#lstm_train_out_bi = np.array(lstm_train_out_bi)

test_lstm_in = [np.array(test_lstm_input) for test_lstm_input in test_lstm_in]
test_lstm_in = np.array(test_lstm_in)

print("train_lstm_in[1].shape",train_lstm_in[1].shape)

# LSTMのモデルを設定
def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(inputs[1].shape)))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model


# ランダムシードの設定
np.random.seed(202)

#print(train_lstm_in)
# 初期モデルの構築
yen_model = build_model(train_lstm_in, output_size=1, neurons = 20)

print(train_lstm_in)
print(train_lstm_in.shape)

print(lstm_test_out)
print(lstm_test_out.shape)

yen_model.summary()


#print(test_lstm_in)




"""
# データを流してフィッティングさせましょう
yen_history = yen_model.fit(train_lstm_in, lstm_train_out, 
                            epochs=50, batch_size=10, verbose=2, shuffle=True)


model_arc_json = yen_model.to_json()
open("fx_predict.json", mode='w').write(model_arc_json)

# 学習済みの重みを保存
yen_model.save_weights('fx_predict_weight.hdf5')

"""

MODEL_ARC_PATH = 'fx_predict.json'
WEIGHTS_PATH = 'fx_predict_weight.hdf5'



model_arc_str = open(MODEL_ARC_PATH).read()
yen_model = model_from_json(model_arc_str)


# 重みの読み込み
yen_model.load_weights(WEIGHTS_PATH)

yen_model.summary()





predict_test = yen_model.predict(test_lstm_in)
predict_test = predict_test.flatten() 

print(predict_test.shape)
print(test_lstm_in.shape)


