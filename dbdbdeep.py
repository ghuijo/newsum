import pandas as pd
import pymysql

conn=None
cur=None

#mariaDB와 연동하는 부분
conn = pymysql.connect(host='newsum.cobmk4nqrt98.ap-northeast-2.rds.amazonaws.com', user='nlsushi', password='nlsushi14', db='newsum_db',charset='utf8', port=3306)
cur = conn.cursor()
sql = "UPDATE article SET recent = false WHERE recent = true"
cur.execute(sql)
conn.commit()
conn.close()

# 데이터 넣기
sql = "INSERT INTO article(company,title,writer,date,link,img,article_origin,article_extractive,category,hashtag,recent) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,true)"

def joonangToDB(joongang_fin):
    for article in joongang_fin:
        conn=None
        cur=None
        conn = pymysql.connect(host='newsum.cobmk4nqrt98.ap-northeast-2.rds.amazonaws.com', user='nlsushi', password='nlsushi14', db='newsum_db',charset='utf8', port=3306)
        cur = conn.cursor()
        val = ("중앙일보",article[0],article[1],article[2],article[3],article[4],article[6],article[7],article[5], article[8])
        cur.execute(sql,val)
        conn.commit()
        conn.close()
    print("중앙디비완료")

def khanToDB(khan_fin):
    for article in khan_fin:
        conn=None
        cur=None
        conn = pymysql.connect(host='newsum.cobmk4nqrt98.ap-northeast-2.rds.amazonaws.com', user='nlsushi', password='nlsushi14', db='newsum_db',charset='utf8', port=3306)
        cur = conn.cursor()
        val = ("경향신문",article[0],article[1],article[2],article[3],article[4],article[6],article[7],article[5], article[8])
        cur.execute(sql,val)
        conn.commit()
        conn.close()
    print("경향디비완료")

def haniToDB(hani_fin):
    for article in hani_fin:
        conn=None
        cur=None
        conn = pymysql.connect(host='newsum.cobmk4nqrt98.ap-northeast-2.rds.amazonaws.com', user='nlsushi', password='nlsushi14', db='newsum_db',charset='utf8', port=3306)
        cur = conn.cursor()
        val = ("한겨레",article[0],article[1],article[2],article[3],article[4],article[6],article[7],article[5], article[8])
        cur.execute(sql,val)
        conn.commit()
        conn.close()
    print("한겨례디비완료")
