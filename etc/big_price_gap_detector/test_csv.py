import numpy as np
import pandas as pd
import re
import time



upbit_raw = pd.read_csv("upbit.csv", encoding="cp949", names = ["name", "price", "percentage", "exchange_rate"])

upbit = []
for name_val, price_val in zip(upbit_raw["name"].values, upbit_raw["price"].values):
    each_name = re.search(r"[A-z]+", name_val).group(0)
    each_price = re.search(r"0.\d+", price_val).group(0)
    upbit.append((each_name, (float)(each_price)))


binance_raw = pd.read_csv("binance.csv", encoding="cp949", names = ["column1"])
binance = []
tmp = []
for idx, column1_val in enumerate(binance_raw["column1"].values):
    if idx % 3 == 0:
        each_name = re.search(r"[A-z]+", column1_val).group(0)
        tmp.append(each_name)
    elif (idx + 1) % 3 == 0:
        continue
    elif (idx + 2) % 3 == 0:
        each_price = (float)(column1_val)
        tmp.append(each_price)

    if len(tmp) == 2 :
        binance.append(tuple(tmp))
        tmp = []


df_upbit = pd.DataFrame(upbit, columns=["name", "u_price"])
df_binance = pd.DataFrame(binance, columns=["name", "b_price"])

df_total = df_upbit.merge(df_binance, on='name', how="inner")

bitcoin = 20619000
df_total = df_total.assign(difference = (df_total["u_price"].values - df_total["b_price"].values))
df_total = df_total.assign(KRW_per_diff = (df_total["u_price"].values - df_total["b_price"].values) * bitcoin)
print(df_total.sort_values(by=["difference"], ascending=[False]))
# print(df_total["di
# fference"] = df_total["u_price"].values - df_total["b_price"].values)
