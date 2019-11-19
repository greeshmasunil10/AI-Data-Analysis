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

lemmatizer = WordNetLemmatizer()

class datatable:
    __title=""
    __year=""
    __type=""
    __prob=0.0
    
    def __init__(self,a,b,c) :
        self.__title=a
        self.__type=b
        self.__year=c
        
    def settitle(self,t):
        self.__title=t 
    def setprob(self,t):
        self.__prob=t 
    def setyear(self,t):
        self.__year=t 
    def settype(self,t):
        self.__type=t  
    def gettitle(self):
        return self.__title 
    def getprob(self):
        return self.__prob 
    def getyear(self):
        return self.__year 
    def gettype(self):
        return self.__type
    def disp(self):
        return(self.__title,self.__year,self.__type)

class word:
    word=""
    count=0
    count1 =0
    count2 = 0
    count3= 0
    count4 = 0
    prob1 = prob2=prob3=prob4=0
    def __init__(self,a,b,c,d,e) :
        self.word=a
        self.count1=b
        self.count2=c
        self.count3=d
        self.count4=e
    def setfreq1(self,t):
        self.count1=t   
        if(story_count!=0):  
            self.prob1=round(self.count1/story_count, 3)
    def setfreq2(self,t):
        self.count2=t 
        if(askhn_count!=0):  
            self.prob2=round(self.count2/askhn_count, 3)
    def setfreq3(self,t):
        self.count3=t    
        if(showhn_count!=0):   
            self.prob3=round(self.count3/showhn_count, 3)
    def setfreq4(self,t):
        self.count4=t  
        if(poll_count!=0):  
            self.prob4=round(self.count4/poll_count, 3)
    def disp(self):
        return(self.word+"  "+
               str(self.count1)+"  "+
               str(self.count1)+"  "+
               str(self.count1)+"  "+
               str(self.count2) +"  "+
               str(self.prob2)+"  "+
               str(self.count3)+"  "+
               str(self.prob3)+"  "+
               str(self.count4)+"  "+
               str(self.prob4))

pudding=[]
wordlist=[]
train_titles=[]
print("Processing data...")


def lemmatizeData():
    print("Lemmatizing...")
    from collections import Counter
    train_set = train_titles
    freq1 = Counter()
    freq2 = Counter()
    freq3 = Counter()
    freq4 = Counter()
    freq=Counter()
    filename="hn2018_2019.csv"
    filename="sample.csv"
    with open(filename,encoding='utf-8') as csvfile:
#         readCSV = pd.read_csv(filename)
        readCSV = csv.reader(csvfile, delimiter=',',)
        print("Segmenting data..")
        for row in readCSV:
            if(row[5][0:4]=='2018'):
                pt = row[3]
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
            else :  
                del row        
    with open(filename,encoding='utf-8') as csvfile:
#         readCSV = pd.read_csv(filename, chunksize=10*8)
        readCSV = csv.reader(csvfile, delimiter=',')
        print("Calculating Frequencies...")
        for row in readCSV:
            if(row[5][0:4]=='2018'):
                word_list = nltk.word_tokenize(row[2].lower())
                title = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
                post_type = row[3]
                print(title)
                if(post_type == "story"):
                    freq1.update(title.split())
                    word_info(freq1,post_type)
                if(post_type == "ask_hn"):
                    freq2.update(title.split())
                    word_info(freq2,post_type)
                if(post_type == "show_hn"):
                    freq3.update(title.split())
                    word_info(freq3,post_type)
                if(post_type == "poll"):
                    freq4.update(title.split())
                    word_info(freq4,post_type)
#                 pud = datatable(title,post_type,row[5][0:4])
#                 pudding.append(pud)
#                 train_titles.append(title)
#     with open(filename,encoding='utf-8') as csvfile:
#         readCSV = csv.reader(csvfile, delimiter=',')
#         print("Calculating Conditional Probabilities...")
#         for row in readCSV:
#             if(row[5][0:4]=='2018'):
#                 word_list = nltk.word_tokenize(row[2].lower())
#                 title = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
#                 post_type = row[3]
#                 if(post_type == "story"):
#                 if(post_type == "ask_hn"):
#                 if(post_type == "show_hn"):
#                 if(post_type == "poll"):
def word_info(freq,post_type):
    i=0
    for k, v in dict(freq).items():
        prob=round(v/sum(freq.values()), 4)
        i+=1
        word1 = word(k,0,0,0,0)
        if(post_type == "story"):
            word1.setfreq1(v)
        if(post_type == "ask_hn"):
            word1.setfreq2(v)
        if(post_type == "show_hn"):
            word1.setfreq3(v)
        if(post_type == "poll"):
            word1.setfreq4(v)
        found = False    
        for w in wordlist:
            if(w.word==k):
                wordlist[wordlist.index(w)] = word1
                found = True
        if(found==False):
            wordlist.append(word1)        
#         print("\n",i,k, v,prob,end="")

story_count =  askhn_count = showhn_count= poll_count = 0

lemmatizeData()
print("Saving information in file..")
i=0
f = open("model-2018.txt", "w")
for w in wordlist:
    i+=1
    f.write(str(i)+'  '+w.disp()+'\n')
#     print(str(i)+'  '+w.disp()+'\n')
f.close()    
    
print("End of Process!")
print("Check file for output ")
# create_vocab()
# print(story_count,askhn_count,showhn_count,poll_count)

