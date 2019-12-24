#!/usr/bin/python3

from selenium import webdriver
from time import sleep
from getpass import getpass
import convolution
from PIL import Image
import numpy as np

student_id = input('student id: ')
password = getpass('password: ')

driver = webdriver.Chrome()
driver.get("https://e3new.nctu.edu.tw/login/")

driver.save_screenshot("screenshot.png")

img = Image.open("screenshot.png")
img = img.crop((280, 440,490,500))
img.save('screenshot.png')
token = '{:04d}'.format(convolution.imgPathToNumber('screenshot.png'))

box = driver.find_element_by_xpath("(//input[@id='username'])[2]")
box.clear()
box.send_keys(student_id)

box = driver.find_element_by_xpath("(//input[@id='password'])[2]")
box.clear()
box.send_keys(password)

box = driver.find_element_by_xpath("(//input[@name='captcha_code'])[2]")
box.clear()
box.send_keys(token)

driver.find_element_by_id("loginbtn-tablet").click()
