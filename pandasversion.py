# -----------------------------------------------
# Project 2
# Written by Greeshma Sunil
# For COMP 6721 Section F FJ(1778) - Fall 2019
# -----------------------------------------------
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer 
import time
import os
from collections import Counter
                                      
start = time.time()
lemmatizer = WordNetLemmatizer()
filename="hn2018_2019.csv"
filename="sample100.csv"
smooth=0.5
wordlist=[]
wordchecklist=[]
freq1 = Counter()
freq2 = Counter()
freq3 = Counter()
freq4 = Counter()
freq = Counter()
global df

class word:
    word=""
    count1 =0
    count2 = 0
    count3= 0
    count4 = 0
    prob1 = 0
    prob2=0
    prob3=0
    prob4=0
    def __init__(self,a) :
        self.word=a
        self.count1=0
        self.count2=0
        self.count3=0
        self.count4=0
        if(storycount!=0):self.prob1=round(smooth/(storycount+vocabsize*smooth),7)
        if(askcount!=0):self.prob2=round(smooth/(askcount+vocabsize*smooth),7)
        if(showcount!=0):self.prob3=round(smooth/(showcount+vocabsize*smooth),7)
        if(pollcount!=0):self.prob4=round(smooth/(pollcount+vocabsize*smooth),7)

    def setfreq(self,t,b,c,d):
        if(t!=-1): self.count1=t   
        if(b!=-1):self.count2=b   
        if(c!=-1): self.count3=c   
        if(d!=-1): self.count4=d  
        if(storycount!=0 and t!=-1):
            self.prob1=round((smooth+t)/(storycount+vocabsize*smooth),3)
        if(askcount!=0 and b!=-1): 
            self.prob2=round((smooth+b)/(askcount+vocabsize*smooth),3)
        if(showcount!=0 and c!=-1): 
            self.prob3=round((smooth+c)/(showcount+vocabsize*smooth),3)
        if(pollcount!=0 and d!=-1): 
            self.prob4=round((smooth+d)/(pollcount+vocabsize*smooth),3)
            
    def disp(self):
        return(self.word+"  "+
               str(self.count1)+"  "+
               str(self.prob1)+"  "+
               str(self.count2) +"  "+
               str(self.prob2)+"  "+
               str(self.count3)+"  "+
               str(self.prob3)+"  "+
               str(self.count4)+"  "+
               str(self.prob4))

def train_data():
    df =pd.read_csv(filename,encoding='ISO-8859-1')
    global storycount, askcount,showcount,pollcount
    print("Segmenting data..")
    for i in range(df.shape[0]):
        if(df['Created At'][i][0:4]=='2018'):
            title = ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(df['Title'][i].lower())])
            freq.update(title.split())
            if(df['Post Type'][i] == "story"):
                freq1.update(title.split())
            if(df['Post Type'][i] == "ask_hn"):
                freq2.update(title.split())
            if(df['Post Type'][i] == "show_hn"):
                freq3.update(title.split())
            if(df['Post Type'][i] == "poll"):
                freq4.update(title.split())
    storycount=len(freq1.values())
    askcount=len(freq2.values())
    showcount=len(freq3.values())
    pollcount=len(freq4.values())
    global vocabsize
    vocabsize=len(freq.values())
    print("Calculating probabilities...")
    j=0
    for i in range(df.shape[0]):
        if(df['Created At'][i][0:4]=='2018'):
            if(df['Post Type'][i] == "story"):
                update_word_frequency(freq1,df['Post Type'][i])
            if(df['Post Type'][i] == "ask_hn"):
                update_word_frequency(freq2,df['Post Type'][i])
            if(df['Post Type'][i] == "show_hn"):
                update_word_frequency(freq3,df['Post Type'][i])
            if(df['Post Type'][i] == "poll"):
                update_word_frequency(freq4,df['Post Type'][i])
        j+=1
        if(j%100==0):
            print("...")
    print("Most common word in story:",freq1.most_common(1)) 

def update_word_frequency(freq,post_type):
    i=1
    for wordname, wordcount in dict(freq).items():
        word1=next((x for x in wordlist if x.word == wordname), None)
        if(word1==None):
            word1 = word(wordname)
        if(post_type == "story"):
            word1.setfreq(wordcount,-1,-1,-1)
        if(post_type == "ask_hn"):
            word1.setfreq(-1,wordcount,-1,-1)
        if(post_type == "show_hn"):
            word1.setfreq(-1,-1,wordcount,-1)
        if(post_type == "poll"):
            word1.setfreq(-1,-1,-1,wordcount)
        if(wordname in wordchecklist):
                wordlist[wordchecklist.index(wordname)] = word1
        else:
            wordlist.append(word1)
            wordchecklist.append(wordname)  
#         print("\n",i,wordname, wordcount,end="")      
        i+=1    

def test_data():
    for i in range(df.shape[0]):
        if(df['Created At'][i][0:4]=='2019'):
            title = ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(df['Title'][i].lower())])
            freq.update(title.split())
            if(df['Post Type'][i] == "story"):
                freq1.update(title.split())
            if(df['Post Type'][i] == "ask_hn"):
                freq2.update(title.split())
            if(df['Post Type'][i] == "show_hn"):
                freq3.update(title.split())
            if(df['Post Type'][i] == "poll"):
                freq4.update(title.split())








print("Processing data...")
train_data()
print("Saving information in file..")
f = open("model-2018.txt", "w",encoding='ISO-8859-1')
wordlist = sorted(wordlist, key=lambda x: x.word, reverse=False)
print("total no of words:",vocabsize)
i=0
for w in wordlist:
#     try:
        i+=1
        f.write(str(i)+'  '+w.disp()+'\n')
#         print((str(i)+'  '+w.disp()))
#     except:
#         continue
f.close()       
print(storycount,askcount,showcount,pollcount)
print(len(freq1.values()),len(freq2.values()),len(freq3.values()),len(freq4.values()))
print("End of Process!")
print("Check file for output ")
end = time.time()
print("Total elapsed time:",round(end - start,1),"seconds")
os.system("notepad.exe model-2018.txt")