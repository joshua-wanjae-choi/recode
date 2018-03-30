#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from selenium.webdriver.support.ui import WebDriverWait


from bs4 import BeautifulSoup
import time
import random
import telegram

# input bot token
bot = telegram.Bot(token="your telegram token")
chat_id =  your chat_id

# Load a Page
driver = webdriver.Firefox(executable_path='./geckodriver')
driver.get('https://upbit.com/exchange?code=CRIX.UPBIT.KRW-BTC')
time.sleep(20)
driver.find_element_by_link_text("BTC").click()
time.sleep(5)

driver.execute_script('''window.open(" ");''')
driver.switch_to_window(driver.window_handles[1])
driver.get("https://www.binance.com/trade.html")
time.sleep(20)


def get_bitcoin(html):
    u_html = html
    u_soup = BeautifulSoup(u_html, 'html.parser')
    span = u_soup.find_all('span', {"class":"first"})
    bitcoin = (span[0].find("strong").get_text())
    bitcoin = (float)(bitcoin.replace(",", ""))
    return bitcoin


def get_upbit_data(html):
    u_html = html
    u_soup = BeautifulSoup(u_html, 'html.parser')

    name = []
    titles = u_soup.find_all('td', {"class":"tit"})
    for title in titles:
        raw_name = title.find('em').get_text()
        name.append(re.search(r"[A-z]+", raw_name).group(0))

    b_price = []
    prices = u_soup.find_all('td', {"class":"price"})
    for price in prices:
        raw_price = price.find('strong').get_text()
        b_price.append((float)(raw_price))

    upbit = [(each_name, each_b_price) for each_name, each_b_price in zip(name, b_price)]
    return upbit


def get_binance_data(html):
    b_html = html
    b_soup = BeautifulSoup(b_html, 'html.parser')

    binance = []
    tmp = []
    table = b_soup.find("div", {"class":"market-con scrollStyle"})
    titles_and_prices = table.find_all("li", {"class":"ng-binding"})
    for idx, title_and_price in enumerate(titles_and_prices) :
            column1_val = title_and_price.get_text()
            if idx % 2 == 0:
                each_name = re.search(r"[A-z]+", column1_val).group(0)
                tmp.append(each_name)
            else:
                each_price = (float)(column1_val)
                tmp.append(each_price)

            if len(tmp) == 2 :
                binance.append(tuple(tmp))
                tmp = []
    return binance




def make_df_result(upbit, binance, bitcoin):
    df_upbit = pd.DataFrame(upbit, columns=["name", "u_price"])
    df_binance = pd.DataFrame(binance, columns=["name", "b_price"])

    df_total = df_upbit.merge(df_binance, on='name', how="inner")

    df_total = df_total.assign(difference_in_BTG = (df_total["u_price"].values - df_total["b_price"].values))
    df_total = df_total.assign(difference_in_KRWofBTG = (df_total["u_price"].values - df_total["b_price"].values) * bitcoin)
    df_total = df_total.sort_values(by=["difference_in_BTG"], ascending=[False])
    return df_total


def send_message_to_user(max_df, min_df, bitcoin, profit_in_BTG, profit_in_KRWofBRG, kim_p_ptg, reverse_p_ptg):
    # # print log
    print("Maximum_kimchi_premium :\n", max_df, "\n")
    print("Minimum_kimchi_premium(Reverse_kimchi_premium) :\n", min_df, "\n")
    print("Expected_profit_per_1_cycle: \n", profit_in_BTG, "BTG\n", profit_in_KRWofBRG, "KRW/BTG\n")
    bot.sendMessage(chat_id=chat_id, text=\
    """
    [  요약  ]
    -최대 김프 :
    종목명 : {}
    차익(BTG) : {}
    차익(KRW/BTG) : {}
    김프수익률 : {:.4}%

    -최소 김프(- 시, 역프):
    종목명 : {}
    차익(BTG) : {}
    차익(KRW/BTG) : {}
    역프수익률 : {:.4}%

    -1 사이클 당 기대이익:
    {} BTG
    {} KRW/BTG

    """.format(max_df['name'].values[0], max_df['difference_in_BTG'].values[0], max_df['difference_in_KRWofBTG'].values[0], kim_p_ptg,\
               min_df['name'].values[0], min_df['difference_in_BTG'].values[0], min_df['difference_in_KRWofBTG'].values[0], reverse_p_ptg,\
               profit_in_BTG, profit_in_KRWofBRG)\
    )



def main():
    while True:
        driver.switch_to_window(driver.window_handles[0])
        time.sleep(5)
        u_html = driver.page_source
        driver.switch_to_window(driver.window_handles[1])
        time.sleep(5)
        b_html = driver.page_source

        bitcoin = get_bitcoin(u_html)
        df = make_df_result(get_upbit_data(u_html), get_binance_data(b_html), bitcoin)
        max_df = df[df['difference_in_BTG'] == df['difference_in_BTG'].max()]
        min_df = df[df['difference_in_BTG'] == df['difference_in_BTG'].min()]

        profit_in_BTG = (max_df["difference_in_BTG"].values - min_df["difference_in_BTG"].values)[0]
        profit_in_KRWofBRG = profit_in_BTG * bitcoin
        kim_p_ptg = (max_df['difference_in_BTG'].values[0]/max_df['u_price'].values[0]) * 100
        reverse_p_ptg = (-min_df['difference_in_BTG'].values[0]/min_df['b_price'].values[0]) * 100

        if (kim_p_ptg + reverse_p_ptg) > 10:
            send_message_to_user(max_df, min_df, bitcoin, profit_in_BTG, profit_in_KRWofBRG, kim_p_ptg, reverse_p_ptg)

        print("log : complete running 1 loop")
        print("Maximum_kimchi_premium :\n", max_df, "\n")
        print("Minimum_kimchi_premium(Reverse_kimchi_premium) :\n", min_df, "\n")
        print("Expected_profit_per_1_cycle: \n", profit_in_BTG, "BTG\n", profit_in_KRWofBRG, "KRW/BTG\n")

        time.sleep(random.randrange(1,3)*10)


if __name__ == "__main__":
    main()
