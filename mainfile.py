# -----------------------------------------------
# Project 1
# Written by Greeshma Sunil
# For COMP 6721 Section F FJ(1778) - Fall 2019
# -----------------------------------------------



import pandas
import statistics
import numpy as np
from builtins import list

  
print("fetching data...")
# hacker_data = pandas.read_csv('hn2018_2019.csv',encoding='utf-8')
hacker_data = pandas.read_csv('sample.csv',encoding='utf-8')

# print(hacker_data,type(hacker_data))
# print(hacker_data.get('Object ID')[2])
# print(hacker_data.get('Title')[2])

lis= list(hacker_data.get('Title'))
for item in lis:
    try:
        print(item)
    except UnicodeEncodeError:
        print("error")
        continue

def create_vocab():
    from collections import Counter
    train_set = lis
    xc = Counter()
    for s in train_set:
        xc.update(s.split())
    print(xc)
    p= dict(xc)
    print("noooooo",type(p),p)
    for k, v in p.items():
         print(k, v)
    print("grees",xc)
    pairs=xc.items()
    print("yoyo",type(pairs))
#     for pp in pairs:
#         print(pp)
    op=list(xc.elements())
    opp=list(xc.keys())
    print(xc.most_common(1))

#     print(type(xc))
#     for it in xc:
#         print(it)

print("passed\n",end="\n")
create_vocab()

