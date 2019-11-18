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

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatize Single Word
print(lemmatizer.lemmatize("bats"))

lis=[]
lisno=[]

class datatable:
    __title=""
    __year=""
    __type=""
    __prob=0.0
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
    
    
# pudding = [datatable()for i in range(100)]   
pudding=[]
train_titles=[]
def lemmatizeData():
    
    with open('sample.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if(row[5][0:4]=='2018'):
                print(row)
                t=row[2].lower()
                word_list = nltk.word_tokenize(t)
                print(word_list)
#                 print("check!!!!!!!!!!!",lemmatizer.lemmatize("books"))
                lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
#                 title=lemmatizer.lemmatize(row[2])
                title=lemmatized_output
                lis.append(t)
                lisno.append(t)
                pud=datatable()
                pud.settitle(title.lower())
                pud.setyear(row[5][0:4])
                pud.settype(row[3])
#                 print(pud.disp())
                pudding.append(pud)
                train_titles.append(title)
        #         print(row[0])
#         print(row[0],row[1],row[2],)
    print("\nprinting")
    for it in pudding:
        print(it.disp())
    print("lists:",lis)
    print("lisno:",train_titles)    

    
print("fetching data...")
# hacker_data = pandas.read_csv('hn2018_2019.csv',encoding='utf-8')
hacker_data = pandas.read_csv('sample.csv',encoding='utf-8')

# print(hacker_data,type(hacker_data))
# print(hacker_data.get('Object ID')[2])
# print(hacker_data.get('Title')[2])

# lis= list(hacker_data.get('Title'))
# for item in lis:
#     try:
#         print(item)
#     except UnicodeEncodeError:
#         print("error")
#         continue

def create_vocab():
    from collections import Counter
    train_set = train_titles
    xc = Counter()
    print("Creating vocabulary...")
    for q in pudding:
        s=q.gettitle()
        xc.update(s.split())
    print(xc)
    p= dict(xc)
    print("counter...",type(p),p)
    i=0
    for k, v in p.items():
         prob=round(v/sum(xc.values()), 3)
         i+=1
         print(i,k, v,prob)
         
#     print("grees",xc)
#     pairs=xc.items()
#     print("yoyo",type(pairs))
# #     for pp in pairs:
# #         print(pp)
#     op=list(xc.elements())
#     opp=list(xc.keys())
#     print(xc.most_common(1))

#     print(type(xc))
#     for it in xc:
#         print(it)

lemmatizeData()
print("titles in 2018",lis)      
print("The end\n",end="\n")
create_vocab()

