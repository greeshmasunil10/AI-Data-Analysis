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
import sys
# import codecs
# sys.stdout = codecs.getwriter('utf8')(sys.stdout)
# sys.stderr = codecs.getwriter('utf8')(sys.stderr)
                                      
lemmatizer = WordNetLemmatizer()

start = time.time()

class word:
    word=""
    count=0
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
        if(story_count!=0):self.prob1=round(0.5/story_count,3)
        if(askhn_count!=0):self.prob2=round(0.5/askhn_count,3)
        if(showhn_count!=0):self.prob3=round(0.5/showhn_count,3)
        if(poll_count!=0):self.prob4=round(0.5/poll_count,3)

    def setfreq(self,t,b,c,d):
        if(t!=-1): self.count1=t   
        if(b!=-1):self.count2=b   
        if(c!=-1): self.count3=c   
        if(d!=-1): self.count4=d  
        if(story_count!=0 and t!=-1): 
            self.prob1=round((t+0.5)/story_count, 3)
        if(askhn_count!=0 and b!=-1): 
            self.prob2=round((b+0.5)/askhn_count, 3)
        if(showhn_count!=0 and c!=-1): 
            self.prob3=round((c+0.5)/showhn_count, 3)
        if(poll_count!=0 and d!=-1): 
            self.prob4=round((d+0.5)/poll_count, 3)
            
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
    filename="sample100.csv"
    df = pd.read_csv(filename,encoding='ISO-8859-1')
    print("Segmenting data..")
    global story_count, askhn_count,showhn_count,poll_count
    for i in range(df.shape[0]):
        if(df['Created At'][i][0:4]=='2018'):
#             print(df['Title'][i].lower())
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
    print("most common word in story:",freq1.most_common(1)) 

def word_info(freq,post_type):
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

story_count =  askhn_count = showhn_count= poll_count = 0

lemmatizeData()
print("Saving information in file..")
i=0
f = open("model-2018.txt", "w",encoding='utf-8')
wordlist = sorted(wordlist, key=lambda x: x.word, reverse=False)
for w in wordlist:
    try:
        i+=1
        f.write(str(i)+'  '+w.disp()+'\n')
        print((str(i)+'  '+w.disp()))
    except:
        continue

f.close()       
print(story_count,askhn_count,showhn_count,poll_count)
print("End of Process!")
print("Check file for output ")
# df1 = pd.read_fwf("model-2018.txt")
# df1.to_csv('log.csv')
end = time.time()
print("Total elapsed time:",round(end - start,1),"seconds")
