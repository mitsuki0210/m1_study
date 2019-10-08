# coding: UTF-8
import urllib.request, urllib.error
from bs4 import BeautifulSoup
from datetime import datetime
import csv
import time



time_flag = True

top_news_title = ""

f = open('BBC_news_honbun.csv', 'a')
writer = csv.writer(f, lineterminator='\n')
f.close()


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

for latest in div_BBC:   
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

#print(latest_news_string)


def get_article_body(latest_news_string):

    latest_news_string = latest_news_string.find_all("a")


    #print(latest_news_string)
    string_ = latest_news_string[0]
    string_ = string_.get("href")

    top_news_url = "https://www.bbc.com" + string_

    if "ink_location=live-reporting-story" in top_news_url:

        top_news_html = urllib.request.urlopen(top_news_url)
    

        soup_top = BeautifulSoup(top_news_html, "html.parser")
        #print(soup_top)

        div_top = soup_top.find_all("div")
        #print(div_top)

        #print(p_top)
        top_news_body = ""

        for div in div_top:   
        # classの設定がされていない要素は、tag.get("class").pop(0)を行うことのできないでエラーとなるため、tryでエラーを回避する
            try:
                # tagの中からclass="n"のnの文字列を摘出します。複数classが設定されている場合があるので
                # get関数では配列で帰ってくる。そのため配列の関数pop(0)により、配列の一番最初を摘出する
                # <span class="hoge" class="foo">  →   ["hoge","foo"]  →   hoge
                class_ = div.get("class").pop(0)

                # 摘出したclassの文字列にmkc-stock_pricesと設定されているかを調べます
                if class_ in "story-body":
                    #print(div)
                    # mkc-stock_pricesが設定されているのでtagで囲まれた文字列を.stringであぶり出します
                    top_news_body = div
                    # 摘出が完了したのでfor分を抜けます
                    break
            except:
                # パス→何も処理を行わない
                pass
        
        soup = top_news_body
    
        for script in soup(["script", "style"]):
            script.decompose()
    

        text=soup.get_text()

        lines= [line.strip() for line in text.splitlines()]

        text="\n".join(line for line in lines if line)

        pos = text.find('Close share panel')

        article_body = text[pos:]

        return article_body
    
    else:
        return 0


