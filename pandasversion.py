# -----------------------------------------------
# Project 2
# Written by Greeshma Sunil
# For COMP 6721 Section F FJ(1778) - Fall 2019
# -----------------------------------------------



import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer 
import time
from mailcap import show

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

    def setfreq(self,t,b,c,d):
        self.count1=t   
        self.count2=b   
        self.count3=c   
        self.count4=d  
        if(story_count!=0): 
            self.prob1=round((self.count1+1)/story_count, 3)
        if(askhn_count!=0): 
            self.prob2=round((self.count2+1)/story_count, 3)
        if(showhn_count!=0): 
            self.prob3=round((self.count3+1)/story_count, 3)
        if(poll_count!=0): 
            self.prob4=round((self.count4+1)/story_count, 3)
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

pudding=[]
wordlist=[]
wordchecklist=[]
train_titles=[]
print("Processing data...")


def lemmatizeData():
    print("Lemmatizing...")
    from collections import Counter
    freq1 = Counter()
    freq2 = Counter()
    freq3 = Counter()
    freq4 = Counter()
    filename="hn2018_2019.csv"
    filename="sample.csv"
    df = pd.read_csv(filename,encoding='utf-8')
    print("Segmenting data..")
    global story_count, askhn_count,showhn_count,poll_count
    for i in range(df.shape[0]):
        if(df['Created At'][i][0:4]=='2018'):
            if(df['Post Type'][i] == "story"):
                story_count+=1
            if(df['Post Type'][i] == "ask_hn"):
                askhn_count+=1
            if(df['Post Type'][i] == "show_hn"):
                showhn_count+=1
            if(df['Post Type'][i] == "poll"):
                poll_count+=1
    for i in range(df.shape[0]):
        if(df['Created At'][i][0:4]=='2018'):
            title = ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(df['Title'][i].lower())])
            if(df['Post Type'][i] == "story"):
                freq1.update(title.split())
                word_info(freq1,df['Post Type'][i])
            if(df['Post Type'][i] == "ask_hn"):
                freq2.update(title.split())
                word_info(freq2,df['Post Type'][i])
            if(df['Post Type'][i] == "show_hn"):
                freq3.update(title.split())
                word_info(freq3,df['Post Type'][i])
            if(df['Post Type'][i] == "poll"):
                freq4.update(title.split())
                word_info(freq4,df['Post Type'][i])
def word_info(freq,post_type):
    for wordname, wordcount in dict(freq).items():
        word1 = word(wordname,0,0,0,0)
        if(post_type == "story"):
            word1.setfreq(wordcount,0,0,0)
        if(post_type == "ask_hn"):
            word1.setfreq(0,wordcount,0,0)
        if(post_type == "show_hn"):
            word1.setfreq(0,0,wordcount,0)
        if(post_type == "poll"):
            word1.setfreq(0,0,0,wordcount)
        if(wordname in wordchecklist):
                wordlist[wordchecklist.index(wordname)] = word1
        else:
            wordlist.append(word1)
            wordchecklist.append(wordname)        

story_count =  askhn_count = showhn_count= poll_count = 0

lemmatizeData()
print("Saving information in file..")
i=0
f = open("model-2018.txt", "w")
for w in wordlist:
    i+=1
    f.write(str(i)+'  '+w.disp()+'\n')
    print(str(i)+'  '+w.disp())
f.close()    
    
print("End of Process!")
print("Check file for output ")
print(story_count,askhn_count,showhn_count,poll_count)
end = time.time()
print("total elapsed time:",end - start)
