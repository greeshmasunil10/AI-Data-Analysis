# -----------------------------------------------
# Project 1
# Written by Greeshma Sunil
# For COMP 6721 Section F FJ(1778) - Fall 2019
# -----------------------------------------------



import pandas
import statistics
import numpy as np

  
print("fetching data...")
hacker_data = pandas.read_csv('hn2018_2019.csv',encoding='utf-8')
print(hacker_data,type(hacker_data))
print(hacker_data.get('Object ID')[2])
print(hacker_data.get('Title')[2])
lis= list(hacker_data.get('Title'))


# import csv
# with open('hn2018_2019.csv', newline='', encoding='utf-8') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         print(row)
        
# print(lis)
for item in lis:
    try:
        print(item)
    except UnicodeEncodeError:
        print("error")
        continue

print("passed")