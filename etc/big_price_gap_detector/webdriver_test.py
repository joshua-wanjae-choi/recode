from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import csv
import time
import random

driver = webdriver.Firefox(executable_path='./geckodriver')
f = open('economy.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow(['document_id','document_text'])

driver.get('http://english.hankyung.com/economy')
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
count = 1

#currentpage has to be only 1~9 inclusive
currentpage = 1
endpage = 2
for i in range(currentpage+1, endpage+1):
    title = soup.find_all('h3', class_="tit")
    date = soup.find_all('span', class_="date")
    for j,k in zip(title,date):
        t_title = j.get_text()
        wr.writerow([count, t_title])
        count += 1

    if i % 10 == 0:
        driver.find_element_by_link_text("뒤로").click()
        continue
    elif i % 10 == 1:
        continue
    else:
        driver.find_element_by_link_text(str(i)).click()
        time.sleep(random.uniform(1, 1.5))
f.close()
