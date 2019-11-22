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
import time

lemmatizer = WordNetLemmatizer()
start = time.time()

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

def countfreq():
    
    
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
    global story_count,askhn_count, showhn_count,poll_count
    story_count =  askhn_count = showhn_count= poll_count = 0
    with open(filename,encoding='utf-8') as csvfile:
        df = pd.read_csv(filename)
        print("Processing data..")
#         print("Finding story count..")
        story_count= len([ df['Created At'][i]  
                          for i in range(df.shape[0]) 
                          if(df['Created At'][i][0:4] == '2018' 
                             and df['Post Type'][i]=='story')])
#       
        
#         print("Finding story freq..")
        [ freq1.update(' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(df['Title'][i].lower())]).split())
                          for i in range(df.shape[0]) 
                          if(df['Created At'][i][0:4] == '2018' 
                             and df['Post Type'][i]=='story')]
        story_count= [ word_info(freq1,df['Post Type'][i])
                          for i in range(df.shape[0]) 
                          if(df['Created At'][i][0:4] == '2018' 
                             and df['Post Type'][i]=='story')]
#         print("Finding ask freq..")
        [ freq1.update(' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(df['Title'][i].lower())]).split())
                          for i in range(df.shape[0]) 
                          if(df['Created At'][i][0:4] == '2018' 
                             and df['Post Type'][i]=='ask_hn')]
        askhn_count= [ word_info(freq1,df['Post Type'][i])
                          for i in range(df.shape[0]) 
                          if(df['Created At'][i][0:4] == '2018' 
                             and df['Post Type'][i]=='ask_hn')]
#         print("Finding show freq..")
        [ freq1.update(' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(df['Title'][i].lower())]).split())
                          for i in range(df.shape[0]) 
                          if(df['Created At'][i][0:4] == '2018' 
                             and df['Post Type'][i]=='show_hn')]
        showhn_count_count= [ word_info(freq1,df['Post Type'][i])
                          for i in range(df.shape[0]) 
                          if(df['Created At'][i][0:4] == '2018' 
                             and df['Post Type'][i]=='show_hn')]
#         print("Finding poll freq..")
        [ freq1.update(' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(df['Title'][i].lower())]).split())
                          for i in range(df.shape[0]) 
                          if(df['Created At'][i][0:4] == '2018' 
                             and df['Post Type'][i]=='poll')]
        poll_count= [ word_info(freq1,df['Post Type'][i])
                          for i in range(df.shape[0]) 
                          if(df['Created At'][i][0:4] == '2018' 
                             and df['Post Type'][i]=='poll')]
#         print("processing done")

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
print(story_count,askhn_count,showhn_count,poll_count)
end = time.time()
print("total elapsed time:",end - start)
