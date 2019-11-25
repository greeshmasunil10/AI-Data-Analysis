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
from mailcap import show
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.corpus import stopwords
                                      
start = time.time()
lemmatizer = WordNetLemmatizer()
filename="Resources\hn2018_2019.csv"
filename="Resources\sample500.csv"
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
        if(storycount!=0):self.prob1=round(smooth/(0+vocabsize*smooth),7)
        if(askcount!=0):self.prob2=round(smooth/(0+vocabsize*smooth),7)
        if(showcount!=0):self.prob3=round(smooth/(0+vocabsize*smooth),7)
        if(pollcount!=0):self.prob4=round(smooth/(0+vocabsize*smooth),7)
#         if(storycount!=0):self.prob1=round(smooth/(storycount+vocabsize*smooth),7)
#         if(askcount!=0):self.prob2=round(smooth/(askcount+vocabsize*smooth),7)
#         if(showcount!=0):self.prob3=round(smooth/(showcount+vocabsize*smooth),7)
#         if(pollcount!=0):self.prob4=round(smooth/(pollcount+vocabsize*smooth),7)
        
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
            
    def updatefreq(self): 
        if(storycount!=0):
            self.prob1=round((smooth+self.count1)/(storycount+vocabsize*smooth),3)
        if(askcount!=0): 
            self.prob2=round((smooth+self.count2)/(askcount+vocabsize*smooth),3)
        if(showcount!=0): 
            self.prob3=round((smooth+self.count3)/(showcount+vocabsize*smooth),3)
        if(pollcount!=0): 
            self.prob4=round((smooth+self.count4)/(pollcount+vocabsize*smooth),3)
            
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

def tokenise_model(df):  
#     df['PostType'] = df.PostType.map({'story': 0, 'ask_hn': 1, 'show_hn':2, 'post':3})
    df['Title'] = df.Title.map(lambda x: x.lower())
    df['Title'] = df.Title.str.replace('[^\w\s]', '')
    df['Title'] = df['Title'].apply(nltk.word_tokenize)
    lemmatizer = WordNetLemmatizer()
    df['Title'] = df['Title'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
    df['Title'] = df['Title'].apply(lambda x: ' '.join(x))
    return df

def train_data():
    global wordlist
    df =pd.read_csv(filename,encoding='ISO-8859-1')
    global storycount, askcount,showcount,pollcount
    global freq1,freq2,freq3,freq4,freq
    print("Segmenting data..")
    df = df[(df['Created At']<'2019')]
    df= tokenise_model(df)
    df[(df['Post Type']=='story')]['Title'].apply(lambda x: [freq1.update(x.split())])
    df[(df['Post Type']=='ask_hn')]['Title'].apply(lambda x: [freq2.update(x.split())])
    df[(df['Post Type']=='show_hn')]['Title'].apply(lambda x: [freq3.update(x.split())])
    df[(df['Post Type']=='poll')]['Title'].apply(lambda x: [freq4.update(x.split())])
    df['Title'].apply(lambda x: [freq.update(x.split())])
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
    df[(df['Post Type']=='story')]['Post Type'].apply(lambda x: update_word_frequency(freq1,x))
    df[(df['Post Type']=='ask_hn')]['Post Type'].apply(lambda x: update_word_frequency(freq2,x))
    df[(df['Post Type']=='show_hn')]['Post Type'].apply(lambda x: update_word_frequency(freq3,x))
    df[(df['Post Type']=='poll')]['Post Type'].apply(lambda x: update_word_frequency(freq4,x))

    print("Most common Word in story:",freq1.most_common(1)) 
    print("Saving information in file..")
    f = open("Output\model-2018.txt", "w",encoding='ISO-8859-1')
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

def update_word_frequency(freq,post_type):
#     i=1
    global stopwords
    for wordname, wordcount in dict(freq).items():
#         if(wordname not in stopwords):
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
#             print("\n",i,wordname, wordcount,end="")      
#         i+=1    

def update_freq():
    global wordlist
    for it in wordlist:
        it.updatefreq()


def test_data(wordlist,resfile):
    print("Testing Data...")
    df =pd.read_csv(filename,encoding='ISO-8859-1')
    c=0
    f=0
    k=0
    f1 = open(resfile, "w",encoding='ISO-8859-1')
    for i in range(df.shape[0]):
        if(df['Created At'][i][0:4]=='2019'):
            title = ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(df['Title'][i].lower())])
            words=title.split()
            storyscore= askscore = showscore= pollscore=0
            if(storyprior!=0):storyscore=math.log(storyprior,10)
            if(askprior!=0):askscore=math.log(askprior,10)
            if(showprior!=0):showscore=math.log(showprior,10)
            if(pollprior!=0):pollscore=math.log(pollprior,10)
            wordlist
            for word1 in words:
                checkword=next((x for x in wordlist if x.word == word1),  
                                None)
#                 print()
                if(checkword==None):
                    checkword=Word(word1)
#                     listt=[checkword.prob1,checkword.prob2,checkword.prob3,checkword.prob4]
#                     print("not found:",word1,listt)
#                     print(word1,listt.index(max(listt)),max(listt))
                else:None
#                     listt=[checkword.prob1,checkword.prob2,checkword.prob3,checkword.prob4]
#                     print("found",word1,listt)
#                     print(word1,listt.index(max(listt)),max(listt))
#                 print(word1,[checkword.prob1,checkword.prob2,checkword.prob3,checkword.prob4])    
                if(checkword.prob1!=0):storyscore+= math.log(checkword.prob1,10)
                if(checkword.prob2!=0):askscore+= math.log(checkword.prob2,10)
                if(checkword.prob3!=0):showscore+= math.log(checkword.prob3,10)
                if(checkword.prob4!=0):pollscore+= math.log(checkword.prob4,10)
                
            check=[storyscore,askscore,showscore,pollscore]  
              
            check=list(filter(lambda a: a != 0, check) )   
            if(check.index(max(check))==0):
                res="story"
            if(check.index(max(check))==1):
                res="ask_hn"
            if(check.index(max(check))==2):
                res="show_hn"
            if(check.index(max(check))==3):
                res="poll"
#             print(title,max(check),res) 
#             print(check)
#             print(res,df['Post Type'][i],end=":")    
            if(df['Post Type'][i]==res):
                label="right"
                c+=1
            else:
                label="wrong"
                f+=1  
#             print(label)    
#             print()    
            k+=1
            f1.write(str(k)+'  '+title+"  "+
                         res+"  "+
                         str(round(storyscore,7))+"  "+
                         str(round(askscore,7))+"  "+
                         str(round(showscore,7))+"  "+
                         str(round(pollscore,7))+"  "+
                         df['Post Type'][i]+"  "+
                         label+"  "+
                         '\n')                   
                
    print(round(c/(c+f)*100,2),"success!")        
    print(round(f/(c+f)*100,2),"failure!")        

def remove_stopwords():
    stopwords = []
    global freq1,freq2,freq3,freq4,freq,storycount,askcount,showcount,pollcount
    global wordlist
    with open("Resources\\Stopwords.txt", "r") as f:
        for line in f:
            stopwords.extend(line.split())
    wordlist=[w for w in wordlist if w.word not in stopwords]
    for it in stopwords:
        if it in dict(freq1):
            del freq1[it]
    for it in stopwords:
        if it in dict(freq2):
            del freq2[it]
    for it in stopwords:
        if it in dict(freq3):
            del freq3[it]
    for it in stopwords:
        if it in dict(freq4):
            del freq4[it]
    for it in stopwords:
        if it in dict(freq):
            del freq[it]
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
    
    f = open("Output\stopword-model.txt", "w",encoding='ISO-8859-1')
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
    
    return wordlist

    
def filterlength():
    global freq1,freq2,freq3,freq4,freq,storycount,askcount,showcount,pollcount
    global wordlist
    for it in dict(freq1):
        if len(it)<=2 or len(it)>=9:
            del freq1[it]
    for it in dict(freq2):
        if len(it)<=2 or len(it)>=9:
            del freq2[it]
    for it in dict(freq3):
        if len(it)<=2 or len(it)>=9:
            del freq3[it]
    for it in dict(freq4):
        if len(it)<=2 or len(it)>=9:
            del freq4[it]
    for it in dict(freq):
        if len(it)<=2 or len(it)>=9:
            del freq[it]
    wordlist=[w for w in wordlist if len(w.word)>2 and len(w.word)<9]
#     print("________________________________")
#     for it in wordlist:
#         print(it.word,len(it.word))

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
    
    f = open("Output\wordlength-model.txt", "w",encoding='ISO-8859-1')
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
    
    return wordlist

def startcode():
    print("Processing data...")
    train_data()
    print(1/vocabsize)
    print("word count:",storycount,askcount,showcount,pollcount)
    print("prior probabilities:",storyprior,askprior,showprior,pollprior)
    print("Check file for output..\n")
#     print(freq)
    test_data(wordlist,"Output\\baseline-result.txt")
    
    
    print("\nTrying again with stop words...")
    stoplist=remove_stopwords()
    print("size:",vocabsize)
    print("word count:",storycount,askcount,showcount,pollcount)
    print("prior probabilities:",storyprior,askprior,showprior,pollprior)
    print("size:",vocabsize)
    update_freq()
    test_data(stoplist,"Output\\stopword-result.txt")
    
    
    print("\nTrying again with length filter...")
    lis=filterlength()
    print("size:",vocabsize)
    print("word count:",storycount,askcount,showcount,pollcount)
    print("prior probabilities:",storyprior,askprior,showprior,pollprior)
    print("size:",vocabsize)
    update_freq()
    test_data(lis,"Output\\wordlength-result.txt")

    end = time.time()
    print("Total elapsed time:",round(end - start,1),"seconds")
#     os.system("notepad.exe Output\\model-2018.txt")
#     os.system("notepad.exe Output\\baseline-result.txt")
    print("End of Process!")

startcode()

