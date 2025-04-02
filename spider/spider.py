from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import json
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium.webdriver.common.keys import Keys
import random

driver_path = r"chromedriver.exe"
service = Service(driver_path)
driver = webdriver.Chrome(service=service)

driver.get("https://x.com/")
driver.delete_all_cookies()
driver.maximize_window()
t = open("cookie.txt")
cookies = json.load(t)
for cookie in cookies:
    if 'sameSite' in cookie:
        del cookie['sameSite']
    driver.add_cookie(cookie)

driver.refresh()
time.sleep(2)
element = driver.find_element(By.XPATH, "/html/body/div[1]/div/div/div[2]/main/div/div/div/div[2]/div/div[2]/div/div/div/div/div[1]/div/div/div/form/div[1]/div/div/div/div/div[2]/div/input")
element.send_keys("test" + Keys.ENTER)
driver.refresh()

time.sleep(8)
# for i in range(10):
#     scroll_distance = random.randint(300, 600)
#     # 执行滚动操作，水平方向保持0，垂直方向使用随机生成的距离滚动
#     driver.execute_script(f"window.scrollBy(0, {scroll_distance});")
#     time.sleep(0.2)
following = driver.find_element(By.XPATH,"/html/body/div[1]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div/div[1]/div/div[5]/div[1]/a/span[1]/span").text
followers = driver.find_element(By.XPATH,"/html/body/div[1]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div/div[1]/div/div[5]/div[2]/a/span[1]/span").text
print(following, followers)

tweets = driver.find_element(By.XPATH,"/html/body/div[1]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div/section/div/div")
tweets_info_list = []
tweets_text_list = []
for tweet in tweets:
    element = tweet.find_element(By.XPATH,".//html/body/div[1]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div/section/div/div/div[7]/div[1]/div/article/div/div/div[2]/div[2]/div[3]/div/div")
    aria_label = element.get_attribute("aria-label")
    texts = tweet.find_element(By.XPATH,"/html/body/div[1]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div/section/div/div/div[9]/div[1]/div/article/div/div/div[2]/div[2]/div[2]/div")
    for text in texts:
        break
    tweets_info_list.append(aria_label)

    # a="//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div/section/div/div/div[1]/div[1]/div/article/div/div/div[2]/div[2]/div[2]/button/span"
    # b="//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div/section/div/div/div[5]/div/div/article/div/div/div[2]/div[2]/div[2]/button/span"
    # c="//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div/section/div/div/div[5]/div/div/article/div/div/div[2]"
a = input()