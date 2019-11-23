# -----------------------------------------------
# Project 2
# Written by Greeshma Sunil
# For COMP 6721 Section F FJ(1778) - Fall 2019
# -----------------------------------------------
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import MWETokenizer
import time
import os
from collections import Counter
import math
                                      
start = time.time()
lemmatizer = WordNetLemmatizer()
filename="hn2018_2019.csv"
filename="sample100.csv"
smooth=0.5
global wordlist

wordlist=[]
wordchecklist=[]
freq1 = Counter()
freq2 = Counter()
freq3 = Counter()
freq4 = Counter()
freq = Counter()
global df

class Word:
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
        
    def getword(self):
        return Word
        
    def getprob(self,a):
        if(a==1):
            return self.prob1
        if(a==2):
            return self.prob2
        if(a==3):
            return self.prob3
        return self.prob4
        

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
    global df
    df =pd.read_csv(filename,encoding='ISO-8859-1')
    global storycount, askcount,showcount,pollcount
    print("Segmenting data..")
    for i in range(df.shape[0]):
        if(df['Created At'][i][0:4]=='2018'):
            tokenizer = MWETokenizer()
#             tokenizer.tokenize('Something about the west wing'.split())
            title = ' '.join([lemmatizer.lemmatize(w) for w in 
#                               tokenizer.tokenize(df['Title'][i].lower().split())])
                            nltk.word_tokenize(df['Title'][i].lower())])
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
    global storyprior,askprior,showprior,pollprior
    storyprior=storycount/vocabsize
    askprior=askcount/vocabsize
    showprior=showcount/vocabsize
    pollprior=pollcount/vocabsize
    
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
    print("Most common Word in story:",freq1.most_common(1)) 

def update_word_frequency(freq,post_type):
    i=1
    for wordname, wordcount in dict(freq).items():
        word1=next((x for x in wordlist if x.word == wordname), None)
        if(word1==None):
            word1 = Word(wordname)
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
    c=0
    f=0
    for i in range(df.shape[0]):
        if(df['Created At'][i][0:4]=='2019'):
            title = ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(df['Title'][i].lower())])
            words=title.split()
            storyscore= askscore = showscore= pollscore=0
            if(storyprior!=0):storyscore=math.log(storyprior,10)
            if(askprior!=0):askscore=math.log(askprior,10)
            if(showprior!=0):showscore=math.log(showprior,10)
            if(pollprior!=0):pollscore=math.log(pollprior,10)
            global wordlist
            for word1 in words:
                checkword=next((x for x in wordlist if x.word == word1),  
                                Word(word1))
                if(checkword.prob1!=0):storyscore+= math.log(checkword.prob1,10)
                if(checkword.prob2!=0):askscore+= math.log(checkword.prob2,10)
                if(checkword.prob3!=0):showscore+= math.log(checkword.prob3,10)
                if(checkword.prob4!=0):pollscore+= math.log(checkword.prob4,10)
            check=[storyscore,askscore,showscore]    
            print(title,max(check)) 
            if(check.index(max(check))==0):
                res="story"
            if(check.index(max(check))==1):
                res="ask_hn"
            if(check.index(max(check))==2):
                res="show_hn"
            if(check.index(max(check))==3):
                res="poll"
            print(res,df['Post Type'][i],end=":")    
            if(df['Post Type'][i]==res):
                print("right")
                c+=1
            else:
                print("wrong")  
                f+=1  
            print()    
    print(round(c/(c+f)*100,2),"success!")        
    print(round(f/(c+f)*100,2),"failure!")        


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
print("word count:",storycount,askcount,showcount,pollcount)
print("prior probabilities:",storyprior,askprior,showprior,pollprior)
print("Check file for output..\n")
end = time.time()
test_data()
print("Total elapsed time:",round(end - start,1),"seconds")
os.system("notepad.exe model-2018.txt")
print("End of Process!")