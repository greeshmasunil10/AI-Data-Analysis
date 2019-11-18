# -----------------------------------------------
# Project 2
# Written by Greeshma Sunil
# For COMP 6721 Section F FJ(1778) - Fall 2019
# -----------------------------------------------



import pandas
import statistics
import numpy as np
from builtins import list
import csv
from attr import __title__
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus.reader import wordlist

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatize Single Word
# print(lemmatizer.lemmatize("bats"))

lis=[]
lisno=[]

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
        return(self.word,self.count1,self.prob1,
               self.count2,self.prob2,
               self.count3,self.prob3,
               self.count4,self.prob4)

pudding=[]
wordlist=[]
train_titles=[]
print("fetching data...")


def lemmatizeData():
    from collections import Counter
    train_set = train_titles
    freq1 = Counter()
    freq2 = Counter()
    freq3 = Counter()
    freq4 = Counter()
    freq=Counter()
    with open('sample.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
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
    with open('sample.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if(row[5][0:4]=='2018'):
#                 print("%%%%%%%%%%%%%%%%%%%%%%%%%%...:",row)
                word_list = nltk.word_tokenize(row[2].lower())
                title = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
                post_type = row[3]
#                 print("\n********",post_type,":",title)    
                if(post_type == "story"):
                    freq1.update(title.split())
                if(post_type == "ask_hn"):
                    freq2.update(title.split())
                if(post_type == "show_hn"):
                    freq3.update(title.split())
                if(post_type == "poll"):
                    freq4.update(title.split())
                freq.update(title.split())

                pud = datatable(title,post_type,row[5][0:4])
                pudding.append(pud)
                train_titles.append(title)
    with open('sample.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if(row[5][0:4]=='2018'):
                word_list = nltk.word_tokenize(row[2].lower())
                title = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
                post_type = row[3]
                if(post_type == "story"):
                    word_info(freq1,post_type)
                if(post_type == "ask_hn"):
#                     print("insideeeeeeeee")
                    word_info(freq2,post_type)
                if(post_type == "show_hn"):
                    word_info(freq3,post_type)
                if(post_type == "poll"):
                    word_info(freq4,post_type)
#                 word_info(freq)
def word_info(freq,post_type):
    i=0
    for k, v in dict(freq).items():
        prob=round(v/sum(freq.values()), 4)
        i+=1
        word1 = word(k,0,0,0,0)
        if(post_type == "story"):
            word1.setfreq1(v)
        if(post_type == "ask_hn"):
#             print("************************found",post_type)
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
        
def create_vocab():
    from collections import Counter
    train_set = train_titles
    xc = Counter()
    print("Creating vocabulary...")
    for q in pudding:
        s=q.gettitle()
        story = q.gettype()
        xc.update(s.split())
        print(q.gettype())
#         pudding[pudding.index(q)] = q
    print(xc)
    p= dict(xc)
#     print("counter...",type(p),p)
    i=0
    for k, v in p.items():
         prob=round(v/sum(xc.values()), 3)
         i+=1
         print(i,k, v,prob)

story_count =  askhn_count = showhn_count= poll_count = 0

lemmatizeData()
print("titles in 2018",lis)      
print("The end\n",end="\n")

i=0
for w in wordlist:
    i+=1
    print(i,w.disp())
# create_vocab()
# print(story_count,askhn_count,showhn_count,poll_count)

