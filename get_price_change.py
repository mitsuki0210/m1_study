import urllib.request, urllib.error
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
#import datetime
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytz
import csv
from dateutil.relativedelta import relativedelta
import oandapy



def datetime_change(a_t):
    if(16 <= a_t.hour <= 23):
        a_t -= one_day
        a_t = "{0:%Y/%m/%d %H:%M:%S}".format(a_t)
        return a_t
    else:
        a_t = "{0:%Y/%m/%d %H:%M:%S}".format(a_t)
        return a_t
        

f = open('BBC_news.csv', 'a')
writer = csv.writer(f, lineterminator='\n')
f.close()

csv_file = open("BBC_news.csv", "r", encoding="ms932", errors="", newline="" )
csv_read = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
for row in csv_read:
    day, time = row[0], row[1]
csv_file.close()

new_date, new_time = day.split()

article_time = new_date + ' ' +  time

one_day = timedelta(days=1)
five_minute = timedelta(minutes=5)

article_time = datetime.strptime(article_time, "%Y/%m/%d %H:%M ")

article_time -= five_minute
article_time = datetime_change(article_time)

print(article_time)






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
 

for i in range(1):
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
#print(df[14799:14800])
#print(article_time)
predict_data = df[df['time'] >= article_time]
#print(predict_data)
#print(predict_data.head(4)))
predict_data_15minute = predict_data.head(4)
print(predict_data_15minute)

print(predict_data_15minute.iat[0, 1])
print(predict_data_15minute.iat[3, 1])
predict_price_change = predict_data_15minute.iat[0, 1]  - predict_data_15minute.iat[3, 1]
print(predict_price_change)

"""
split_date = '2019/07/19 20:40:00'
train, test = df[df['time'] < split_date], df[df['time']>=split_date]
del train['time']
del test['time']
"""

"""

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