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

csv_data = "BBC_news.csv"

def datetime_change(a_t):
    if(16 <= a_t.hour <= 23):
        a_t -= one_day
        a_t = "{0:%Y/%m/%d %H:%M:%S}".format(a_t)
        return a_t
    else:
        a_t = "{0:%Y/%m/%d %H:%M:%S}".format(a_t)
        return a_t
        
def make_csv(csv_name):
    f = open(csv_data, 'a')
    writer = csv.writer(f, lineterminator='\n')
    f.close()

def get_article_time(csv_data):
    csv_file = open(csv_data, "r", encoding="ms932", errors="", newline="" )
    csv_read = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)

    
    for row in csv_read:
        #print(row)
        day, time = row[0], row[1]
    
    csv_file.close()
    #print("aaaaaa")
   
    
    new_date, new_time = day.split()

    article_time = new_date + ' ' +  time

    one_day = timedelta(days=1)
    five_minute = timedelta(minutes=5)

    article_time = datetime.strptime(article_time, "%Y/%m/%d %H:%M ")

    article_time -= five_minute
    article_time = datetime_change(article_time)

    return article_time





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


def get_oanda_data():
    ACCOUNT_ID = "1401673"
    ACCESS_TOKEN = "88cb1d105fa6e388042747e4bfea5943-a4d0a01845fd3924f3aaf6ab10fce462"
    
    oanda = oandapy.API(environment="practice", access_token=ACCESS_TOKEN)
    

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


    df = res2[['time','closeAsk']]
    df.columns = ['time','close']

    return df


def get_price_change(df, article_time):
    predict_data = df[df['time'] >= article_time]

    predict_data_15minute = predict_data.head(4)

    predict_price_change = predict_data_15minute.iat[0, 1]  - predict_data_15minute.iat[3, 1]

    if predict_price_change > 0.5:
        level = 0
    if 0 <= predict_price_change <= 0.5:
        level = 1
    if -0.5 <= predict_price_change < 0:
        level = 2
    if predict_price_change < -0.5:
        level = 3
    return level

make_csv(csv_data)
article_time = get_article_time(csv_data)
df = get_oanda_data()
predict_price_change = get_price_change(df, article_time)

print(predict_price_change)