# coding: UTF-8
import urllib.request, urllib.error
from bs4 import BeautifulSoup
from datetime import datetime
import csv
import time


time_flag = True

# 永久に実行させます
while True:
    # 時間が59分以外の場合は58秒間時間を待機する
    #if (datetime.now().minute != 59 and datetime.now().minute != 21 and datetime.now().minute != 29 and datetime.now().minute != 48 and datetime.now().minute != 16):
        # 59分ではないので1分(58秒)間待機します(誤差がないとは言い切れないので58秒です)
    time.sleep(58)
        #continue

    # csvを追記モードで開きます→ここでcsvを開くのはファイルが大きくなった時にcsvを開くのに時間がかかるためです
    f = open('BBC_news.csv', 'a')
    writer = csv.writer(f, lineterminator='\n')

    # 59分になりましたが正確な時間に測定をするために秒間隔で59秒になるまで抜け出せません
    while datetime.now().second != 59:
            # 00秒ではないので1秒待機
            time.sleep(1)
    # 処理が早く終わり二回繰り返してしまうのでここで一秒間待機します
    time.sleep(1)

    # csvに記述するレコードを作成します
    csv_list = []



    # 現在の時刻を年、月、日、時、分、秒で取得します
    time_ = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    # 1カラム目に時間を挿入します
    csv_list.append(time_)

    # アクセスするURL
    url = "https://www.bbc.com/news/uk"

    # URLにアクセスする htmlが帰ってくる → <html><head><title>経済、株価、ビジネス、政治のニュース:日経電子版</title></head><body....
    html = urllib.request.urlopen(url)

    # htmlをBeautifulSoupで扱う
    soup_BBC = BeautifulSoup(html, "html.parser")

    # span要素全てを摘出する→全てのspan要素が配列に入ってかえされます→[<span class="m-wficon triDown"></span>, <span class="l-h...
    #p_BBC = soup_BBC.find_all("p")
    #span_BBC = soup_BBC.find_all("span")
    div_BBC = soup_BBC.find_all("div")

    # print時のエラーとならないように最初に宣言しておきます。
    latest_title_list = []
    latest_time_list = []


    # for分で全てのspan要素の中からClass="mkc-stock_prices"となっている物を探します

    for latest in div_BBC:
        #append(nikkei_heikin_topnews, tag.string)
        
        # classの設定がされていない要素は、tag.get("class").pop(0)を行うことのできないでエラーとなるため、tryでエラーを回避する
        try:
            # tagの中からclass="n"のnの文字列を摘出します。複数classが設定されている場合があるので
            # get関数では配列で帰ってくる。そのため配列の関数pop(0)により、配列の一番最初を摘出する
            # <span class="hoge" class="foo">  →   ["hoge","foo"]  →   hoge
            string_ = latest.get("class").pop(0)
            
        
        

            # 摘出したclassの文字列にmkc-stock_pricesと設定されているかを調べます
            if string_ in "lx-stream":
                # mkc-stock_pricesが設定されているのでtagで囲まれた文字列を.stringであぶり出します
                latest_news_string = latest
                # 摘出が完了したのでfor分を抜けます
                break
        except:
            # パス→何も処理を行わない
            pass


    latest_news_string = latest_news_string.find_all("span")


    for i in range(len(latest_news_string)):
        string_ = latest_news_string[i]
        try:
            string_ = string_.get("class").pop(0)

            if string_ in "lx-stream-post__header-text":
                latest_title_list.append(latest_news_string[i].string)
            elif string_ in "qa-meta-time gs-u-display-inline-block gs-u-mr":
                latest_time_list.append(latest_news_string[i].text)
        
        except:
            pass


    most_latest_news_title = latest_title_list[0]
    most_latest_news_time = latest_time_list[0]

   

    # URLにアクセスする htmlが帰ってくる → <html><head><title>経済、株価、ビジネス、政治のニュース:日経電子版</title></head><body....
    html = urllib.request.urlopen(url)

    # 摘出した日経平均株価を時間とともに出力します。
    print(time_, most_latest_news_time, most_latest_news_title)
    # 2カラム目に記事の時間を記録します
    csv_list.append(most_latest_news_time)
    # 3カラム目に記事のタイトルを記録します
    csv_list.append(most_latest_news_title)
    # csvに追記敷きます
    writer.writerow(csv_list)
    # ファイル破損防止のために閉じます
    f.close()

#---------------------------------------------------------------------------
