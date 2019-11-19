# -----------------------------------------------
# Project 2
# Written by Greeshma Sunil
# For COMP 6721 Section F FJ(1778) - Fall 2019
# -----------------------------------------------



import pandas as pd
import statistics
import numpy as np
from builtins import list
import csv
from attr import __title__
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus.reader import wordlist
from controller import story_count

lemmatizer = WordNetLemmatizer()

story_count =  askhn_count = showhn_count= poll_count = 0
def lemmatizeData():
    print("Lemmatizing...")
    from collections import Counter
    freq1 = Counter()
    freq2 = Counter()
    freq3 = Counter()
    freq4 = Counter()
    freq=Counter()
    filename="hn2018_2019.csv"
    filename="sample.csv"
    with open(filename,encoding='utf-8') as csvfile:
        df = pd.read_csv(filename)
#         for it in df['Created At']:
#             if(it[0:4]=='2019'):
#                 print(it)
#         print(type(df["Title"]))
        print(type(df.shape[0]))
        for i in range(df.shape[0]):
            print(df['Title'][i])
            if(df['Created At'][i][0:4]=='2018'):
                pt = df['Post Type'][i]
                if(pt == "story"):
                    global story_count
                    story_count+=1
                if(pt == "ask_hn"):
                    global askhn_count
                    askhn_count+=1
                if(pt == "show_hn"):
                    global showhn_count
                    showhn_count+=1
                if(pt == "poll"):
                    global poll_count
                    poll_count+=1
        print("count...............",story_count)
        print("count...............",askhn_count)
                              
            

lemmatizeData()