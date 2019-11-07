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


split_date = '2019/09/27 12:45:00'
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


"""

predict_test_bi = []

for i  in range(len(predict_test)):
    if(predict_test[i] - test_lstm_in[i][-1]  >=0):
        predict_test_bi.append(1)
    elif(predict_test[i] - test_lstm_in[i][-1] < 0):
        predict_test_bi.append(0)
predict_test_bi = np.array(predict_test_bi)
print("predict_test_bi",predict_test_bi)

print("predict_test_bi.shape",predict_test_bi.shape)

test_out_bi = []

for i  in range(len(lstm_test_out)):
    if(lstm_test_out[i] - test_lstm_in[i][-1] >=0):
        test_out_bi.append(1)
    elif(lstm_test_out[i] - test_lstm_in[i][-1] < 0):
        test_out_bi.append(0)
test_out_bi = np.array(test_out_bi)

print("test_out_bi",test_out_bi)
print("test_out_bi.shape",test_out_bi.shape)
accuracy_count = 0

for i in range(len(test_out_bi)):
    if(predict_test_bi[i] == test_out_bi[i]):
        accuracy_count += 1

accuracy = accuracy_count / len(test_out_bi) * 100

test_bi_up_count = 0
test_bi_down_count = 0

for i in test_out_bi:
    if(i == 1):
        test_bi_up_count += 1
    elif(i == 0):
        test_bi_down_count += 1
    
print("テストデータの上げのデータ数",test_bi_up_count,"個")
print("テストデータの下げのデータ数",test_bi_down_count,"個")




print("binarry_testdata_accuracy is",accuracy, "%")



predict_train = yen_model.predict(train_lstm_in)
predict_train = predict_train.flatten() 

predict_train_bi = []   

for i  in range(len(predict_train)):
    if(predict_train[i] - train_lstm_in[i][-1] >=0):
        predict_train_bi.append(1)
    elif(predict_train[i] - train_lstm_in[i][-1] < 0):
        predict_train_bi.append(0)
predict_train_bi = np.array(predict_train_bi)
print("predict_train_bi",predict_train_bi)

print("predict_train_bi.shape",predict_train_bi.shape)

train_out_bi = []

for i  in range(len(lstm_train_out)):
    if(lstm_train_out[i] - train_lstm_in[i][-1] >=0):
        train_out_bi.append(1)
    elif(lstm_train_out[i] - train_lstm_in[i][-1] < 0):
        train_out_bi.append(0)
train_out_bi = np.array(train_out_bi)

print("train_out_bi",train_out_bi)
print("train_out_bi.shape",train_out_bi.shape)
accuracy_count = 0

for i in range(len(train_out_bi)):
    if(predict_train_bi[i] == train_out_bi[i]):
        accuracy_count += 1

accuracy = accuracy_count / len(train_out_bi) * 100


train_bi_up_count = 0
train_bi_down_count = 0

for i in train_out_bi:
    if(i == 1):
        train_bi_up_count += 1
    elif(i == 0):
        train_bi_down_count += 1
    
print("訓練データの上げのデータ数",train_bi_up_count,"個")
print("訓練データの下げのデータ数",train_bi_down_count,"個")

print("binarry_traindata_accuracy is",accuracy, "%")


#----------------------------------------------
#predict_test = yen_model.predict(test_lstm_in)
#predict_test = predict_test.flatten() 

predict_test_four = []
threshold = 0.0002 #閾値
print(threshold)

#print(predict_test)

for i  in range(len(predict_test)):
    if(predict_test[i] - test_lstm_in[i][-1] >= threshold):
        predict_test_four.append(0)
    elif(0<= predict_test[i] - test_lstm_in[i][-1] < threshold):
        predict_test_four.append(1)
    elif(-threshold <= predict_test[i] - test_lstm_in[i][-1] < 0):
        predict_test_four.append(2)
    elif(predict_test[i] - test_lstm_in[i][-1] < -threshold):
        predict_test_four.append(3)
predict_test_four = np.array(predict_test_four)
print("predict_test_four",predict_test_four)

print("predict_test_four.shape",predict_test_four.shape)

test_out_four = []

#print(lstm_test_out)

for i  in range(len(lstm_test_out)):
    if(lstm_test_out[i] - test_lstm_in[i][-1] >= threshold):
        test_out_four.append(0)
    elif(0<= lstm_test_out[i] - test_lstm_in[i][-1] < threshold):
        test_out_four.append(1)
    elif(-threshold <= lstm_test_out[i] - test_lstm_in[i][-1] < 0):
        test_out_four.append(2)
    elif(lstm_test_out[i] - test_lstm_in[i][-1] < -threshold):
        test_out_four.append(3)
test_out_four = np.array(test_out_four)

print("test_out_four",test_out_four)
print("test_out_four.shape",test_out_four.shape)
accuracy_count = 0

for i in range(len(test_out_four)):
    if(predict_test_four[i] == test_out_four[i]):
        accuracy_count += 1

accuracy = accuracy_count / len(test_out_four) * 100


test_0_count = 0
test_1_count = 0
test_2_count = 0
test_3_count = 0



for i in test_out_four:
    if(i == 0):
        test_0_count += 1
    elif(i == 1):
        test_1_count += 1
    elif(i == 2):
        test_2_count += 1
    elif(i == 3):
        test_3_count += 1
    
print("テストデータの0のデータ数",test_0_count,"個")
print("テストデータの1のデータ数",test_1_count,"個")
print("テストデータの2のデータ数",test_2_count,"個")
print("テストデータの3のデータ数",test_3_count,"個")


print("four_testdata_accuracy is",accuracy, "%")






#predict_train = yen_model.predict(train_lstm_in)
#predict_train = predict_train.flatten() 

predict_train_four = []

for i  in range(len(predict_train)):
    if(predict_train[i] - train_lstm_in[i][-1] >= threshold):
        predict_train_four.append(0)
    elif(0<= predict_train[i] -  train_lstm_in[i][-1] < threshold):
        predict_train_four.append(1)
    elif(-threshold <= predict_train[i] - train_lstm_in[i][-1] < 0):
        predict_train_four.append(2)
    elif(predict_train[i] - train_lstm_in[i][-1] < -threshold):
        predict_train_four.append(3)
predict_train_four = np.array(predict_train_four)
print("predict_train",predict_train)
print("predict_train_four",predict_train_four)

print("predict_train_four.shape",predict_train_four.shape)

train_out_four = []

for i  in range(len(lstm_train_out)):
    if(lstm_train_out[i] - train_lstm_in[i][-1] >= threshold):
        train_out_four.append(0)
    elif(0<= lstm_train_out[i] - train_lstm_in[i][-1] < threshold):
        train_out_four.append(1)
    elif(-threshold <= lstm_train_out[i] - train_lstm_in[i][-1] < 0):
        train_out_four.append(2)
    elif(lstm_train_out[i] - train_lstm_in[i][-1] < -threshold):
        train_out_four.append(3)
train_out_four = np.array(train_out_four)

print("lstm_train_out",lstm_train_out)
print("train_out_four",train_out_four)
print("train_out_four.shape",train_out_four.shape)
accuracy_count = 0

for i in range(len(train_out_four)):
    if(predict_train_four[i] == train_out_four[i]):
        accuracy_count += 1

accuracy = accuracy_count / len(train_out_four) * 100

train_0_count = 0
train_1_count = 0
train_2_count = 0
train_3_count = 0



for i in train_out_four:
    if(i == 0):
        train_0_count += 1
    elif(i == 1):
        train_1_count += 1
    elif(i == 2):
        train_2_count += 1
    elif(i == 3):
        train_3_count += 1
    
print("訓練データの0のデータ数",train_0_count,"個")
print("訓練データの1のデータ数",train_1_count,"個")
print("訓練データの2のデータ数",train_2_count,"個")
print("訓練データの3のデータ数",train_3_count,"個")

print("four_traindata_accuracy is",accuracy, "%")



"""


#Lot = 0.1
Tsuka = 10000
total_money = 1000000
first_total_money = 1000000
win_count = 0
lose_count = 0
#max_lot = total_money / ((current_late * Tsuka) / 25)
#pal = pfofit and loss
#ppp = Tsuka * 0.01 * max_lot


predict_test = yen_model.predict(test_lstm_in)
predict_test = predict_test.flatten()
Totalpal = 0 
max_win = 0
max_lose = 0
trade_count_list = []
total_money_list = []



for i in range(len(predict_test)-window_len):
    #print(i)
    #print(total_money)
    trade_count_list.append(i)
    max_lot = total_money / ((test['close'][:-window_len].values[i] * Tsuka) / 25)
    ppp = Tsuka * 0.01 * max_lot
    if(predict_test[i] < 0):
        pal = (test['close'][:-window_len].values[i] - test['close'][:-window_len].values[i + window_len]) * ppp
        if(pal >= 0):
            if(pal > max_win):
                max_win = pal
        elif(pal < 0):
            if(pal < max_lose):
                max_lose = pal
        total_money += pal
    elif(predict_test[i] >= 0):
        pal = (test['close'][:-window_len].values[i + window_len] - test['close'][:-window_len].values[i]) * ppp
        if(pal >= 0):
            if(pal > max_win):
                max_win = pal
        elif(pal < 0):
            if(pal < max_lose):
                max_lose = pal
        total_money += pal
    if(pal > 0):
        win_count += 1
    elif(pal < 0):
        lose_count += 1
    total_money_list.append(total_money)
        

up_count = 0
down_count = 0
no_change_count = 0

for i in predict_test:
    if(i > 0):
        up_count += 1
    elif(i < 0):
        down_count += 1
    elif(i == 0):
        no_change_count += 1
    

up_ratio = up_count / len(predict_test)
down_ratio = down_count / len(predict_test)
no_change_ratio = no_change_count / len(predict_test)



print("上げの内訳",up_ratio)
print("下げの内訳",down_ratio)
print("変化なしの内訳",no_change_ratio)
    


print("利益は",total_money,"円です。")
print("勝った回数は",win_count,"回です。")
print("負けた回数は",lose_count,"回です。")

print("勝率は",(win_count / (len(predict_test)-window_len)),"％です。")
print("負け率は",(lose_count / (len(predict_test)-window_len)),"％です。")

print("最大勝ちは",max_win,"円")
print("最大負けは",max_lose,"円")


print(len(trade_count_list))
print(len(total_money_list))

total_money_per = []

for val in total_money_list:
    per = val / first_total_money * 100
    total_money_per.append(per)

fig, ax1 = plt.subplots(1,1)
ax1.plot(trade_count_list, total_money_per)
#fig_money.set_title('Line Plot of Sine and Cosine Between -2\pi and 2\pi')
ax1.set_xlabel("Number of trades")
ax1.set_ylabel("Total assets increase rate (%)")
plt.show()
