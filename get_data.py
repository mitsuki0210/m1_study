import urllib.request, urllib.error
from bs4 import BeautifulSoup
import datetime
import csv
from dateutil.relativedelta import relativedelta


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

one_day = datetime.timedelta(days=1)

article_time = datetime.datetime.strptime(article_time, "%Y/%m/%d %H:%M ")

print(datetime_change(article_time))
