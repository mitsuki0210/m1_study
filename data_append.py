import csv
from get_price_change import get_price_change, get_oanda_data, date_string, iso_jp, datetime_change, get_article_time
import urllib.request, urllib.error
from bs4 import BeautifulSoup
#import datetime
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytz
import csv
from dateutil.relativedelta import relativedelta
import oandapy


csv_data = "BBC_news.csv"

def make_data(csv_data):
    df = get_oanda_data()
    list_ = []
    count = 0
    with open(csv_data, 'r', newline='', encoding='utf-8') as f:
        r = csv.reader(f)  # CSVファイルを読み込んでReaderオブジェクトを生成
        for l in r:
            list__ = []
            day, time = l[0], l[1]
            new_date, new_time = day.split()
            article_time = new_date + ' ' +  time
            one_day = timedelta(days=1)
            five_minute = timedelta(minutes=5)
            article_time = datetime.strptime(article_time, "%Y/%m/%d %H:%M ")
            article_time -= five_minute
            article_time = datetime_change(article_time)
            predict_price_change = get_price_change(df, article_time)
            list__.append(l[3])
            list__.append(predict_price_change)
            list_.append(list__)
        
    return list_


"""
for i in list_:
    day, time = i[0], i[1]
    new_date, new_time = day.split()
    article_time = new_date + ' ' +  time
    one_day = timedelta(days=1)
    five_minute = timedelta(minutes=5)
    article_time = datetime.strptime(article_time, "%Y/%m/%d %H:%M ")
    article_time -= five_minute
    article_time = datetime_change(article_time)
    predict_price_change = get_price_change(df, article_time)
"""


    

    

        