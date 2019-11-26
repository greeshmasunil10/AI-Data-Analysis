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
from builtins import input
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import accuracy_score, recall_score,\
    precision_score, f1_score
from test.test_lzma import INPUT
                                      

lemmatizer = WordNetLemmatizer()
filename="Resources\hn2018_2019.csv"
filename="Resources\sample100.csv"
global wordlist

wordlist=[]
wordchecklist=[]
freq1 = Counter()
freq2 = Counter()
freq3 = Counter()
freq4 = Counter()
freq = Counter()
global df

# '''
# The class word is used for easy storage of probability and frequency information
# of each word in the vocabulary
# General notation: 1-story 2-ask_hn 3-show_hn 4-poll
# This helps in easy retrieval of probabilities in testing stage
# word class Member functions help to update frequencies in experiments stage
# prob1,prob2,prob3,prob4 are the smoothed conditional probabilities
# the list of words objects in vocabulary is stored in 'wordlist'
# '''
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
        smooth=0.5
        if(storycount!=0):self.prob1=round(smooth/(storycount+vocabsize*smooth),7)
        if(askcount!=0):self.prob2=round(smooth/(askcount+vocabsize*smooth),7)
        if(showcount!=0):self.prob3=round(smooth/(showcount+vocabsize*smooth),7)
        if(pollcount!=0):self.prob4=round(smooth/(pollcount+vocabsize*smooth),7)
            
    def setfreq(self,t,b,c,d):
        smooth=0.5
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
        smooth=0.5
        if(storycount!=0):
            self.prob1=round((smooth+self.count1)/(storycount+vocabsize*smooth),3)
        if(askcount!=0): 
            self.prob2=round((smooth+self.count2)/(askcount+vocabsize*smooth),3)
        if(showcount!=0): 
            self.prob3=round((smooth+self.count3)/(showcount+vocabsize*smooth),3)
        if(pollcount!=0): 
            self.prob4=round((smooth+self.count4)/(pollcount+vocabsize*smooth),3)
        
    def updatesmooth(self,smooth): 
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

# ''' 
# This function does all preprocessing at once returns the data frame as whole.
# This is usefull since it is used for both training and testing model
# '''
def preprocess_model(df):  
    df['Title'] = df.Title.map(lambda x: x.lower())
    df['Title'] = df.Title.str.replace('[^\w\s]', '')
    df['Title'] = df['Title'].apply(nltk.word_tokenize)
    lemmatizer = WordNetLemmatizer()
    df['Title'] = df['Title'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
    df['Title'] = df['Title'].apply(lambda x: ' '.join(x))
    return df

# '''
# List comprehension statements is used for better perfomance
# I have also reduced memory footprint in most statements for better speed
# storyprior, askprior,showprior,pollprior and the prior probabilities of each class
# 
# The frequencies ore calculated for each class
# The vocabulary size of each class and total vocabulary size is calculated here
# '''
def train_data():
    global wordlist, storycount, askcount,showcount,pollcount, freq1,freq2,freq3,freq4,freq
    global storyprior,askprior,showprior,pollprior, vocabsize
    df =pd.read_csv(filename,encoding='ISO-8859-1')
    print("\nProcessing data..")
    df = df[(df['Created At']<'2019')]
    df= preprocess_model(df)
    df[(df['Post Type']=='story')]['Title'].apply(lambda x: [freq1.update(x.split())])
    df[(df['Post Type']=='ask_hn')]['Title'].apply(lambda x: [freq2.update(x.split())])
    df[(df['Post Type']=='show_hn')]['Title'].apply(lambda x: [freq3.update(x.split())])
    df[(df['Post Type']=='poll')]['Title'].apply(lambda x: [freq4.update(x.split())])
    df['Title'].apply(lambda x: [freq.update(x.split())])
    storycount=len(freq1.values())
    askcount=len(freq2.values())
    showcount=len(freq3.values())
    pollcount=len(freq4.values())
    vocabsize=len(freq.values())
    storyprior=storycount/vocabsize
    askprior=askcount/vocabsize
    showprior=showcount/vocabsize
    pollprior=pollcount/vocabsize
    
    print("Calculating probabilities...")
    df[(df['Post Type']=='story')]['Post Type'].apply(lambda x: update_word_frequency(freq1,x))
    df[(df['Post Type']=='ask_hn')]['Post Type'].apply(lambda x: update_word_frequency(freq2,x))
    df[(df['Post Type']=='show_hn')]['Post Type'].apply(lambda x: update_word_frequency(freq3,x))
    df[(df['Post Type']=='poll')]['Post Type'].apply(lambda x: update_word_frequency(freq4,x))

    print("Saving Model to file..")
    f = open("Output\model-2018.txt", "w",encoding='ISO-8859-1')
    f2 = open("Output\\vocabulary.txt", "w",encoding='ISO-8859-1')
    wordlist = sorted(wordlist, key=lambda x: x.word, reverse=False)
    i=0
    for w in wordlist:
        i+=1
        f.write(str(i)+'  '+w.disp()+'\n')  
        f2.write(w.word+'\n')
    f.close()  
    f2.close()  

# '''
# This update the frequencies and probabilities to the word-class
# '''
def update_word_frequency(freq,post_type):
    global stopwords
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


def update_freq():
    global wordlist
    for it in wordlist:
        it.updatefreq()
def update_smooth(val):
    global wordlist
    for it in wordlist:
        it.updatesmooth(val)


# '''
# This calculates the score for each class
# score is stored in storyscore,askcore,showscore and pollscore.
# Predicts the Post Type based on score
# '''
def test_data(wordlist,resfile):
    print("--------------------------------------------------------")     
    print("Testing Data...")
    df =pd.read_csv(filename,encoding='ISO-8859-1')
    c=0
    f=0
    k=0
    y_true=[]
    y_pred=[]
    f1 = open(resfile, "w",encoding='ISO-8859-1')
#     df = df[(df['Created At']>='2019')]
    df= preprocess_model(df)
    for i in range(df.shape[0]):
        if(df['Created At'][i][0:4]=='2019'):
#             title = ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(df['Title'][i].lower())])
            title= df['Title'][i]
            words=title.split()
            global storyprior,showprior,askprior,pollprior
            storyscore= askscore = showscore= pollscore=0.0
            if(storyprior!=0):storyscore=math.log(storyprior,10)
            if(askprior!=0):askscore=math.log(askprior,10)
            if(showprior!=0):showscore=math.log(showprior,10)
            if(pollprior!=0):pollscore=math.log(pollprior,10)
            wordlist
            for word1 in words:
                checkword=next((x for x in wordlist if x.word == word1),  
                                None)
                if(checkword==None):
                    None
#                     checkword=Word(word1)
                else:
                    if(checkword.prob1!=0):storyscore+= math.log(checkword.prob1,10)
                    if(checkword.prob2!=0):askscore+= math.log(checkword.prob2,10)
                    if(checkword.prob3!=0):showscore+= math.log(checkword.prob3,10)
                    if(checkword.prob4!=0):pollscore+= math.log(checkword.prob4,10)
                
            check=[storyscore,askscore,showscore,pollscore]  
#             print("max",max(check))    
                  
            check=list(filter(lambda a: a != 0, check) )   
            if(check==[]):
                print("Vocabulary is too small.. Cannot predict Post Type!")
                return 0
            if(check.index(max(check))==0):
                res="story"
            if(check.index(max(check))==1):
                res="ask_hn"
            if(check.index(max(check))==2):
                res="show_hn"
            if(check.index(max(check))==3):
                res="poll"
            if(df['Post Type'][i]==res):
                label="right"
                c+=1
            else:
                label="wrong"
                f+=1  
            y_true.append(df['Post Type'][i])
            y_pred.append(res)
            
            
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
    
#     print(y_true)
#     print(y_pred) 
    print("--------------------------------------------------------")     
    print('Results')     
    print("confusion matrix:-\n",confusion_matrix(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    print("tn,fp,fn,tp :",tn, fp, fn, tp)
    print("Accuracy:",accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='micro') )
    print("Recall:", recall_score(y_true, y_pred, average='macro')  )
    print("F1 measure:", f1_score(y_true, y_pred, average='micro')  )
    print("This model is ",str(round(c/(c+f)*100,2))+"%","accurate") 
    print("--------------------------------------------------------")     
    return round(c/(c+f)*100,2)       
# ''' 
# This performs the functions for testing after removing stop words
# Removes words from wordlist that are in stop list
# updates frequencies and probabilities s
# '''
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
    i=0
    for w in wordlist:
        i+=1
        f.write(str(i)+'  '+w.disp()+'\n')  
    f.close() 
    
    return wordlist

# ''' 
# This performs the functions for testing after filtering length
# Remove words from wordlist after filtering
# updates frequencies and probabilities 
# '''   
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
    i=0
    for w in wordlist:
        i+=1
        f.write(str(i)+'  '+w.disp()+'\n')  
    f.close()
    
    return wordlist

# ''' 
# This performs the functions for testing after filtering length
# Remove words from wordlist after removing infrequent words
# updates frequencies and probabilities 
# ''' 
def freqfilter(n):
    global freq1,freq2,freq3,freq4,freq,storycount,askcount,showcount,pollcount
    global wordlist
    for it,ik in dict(freq1).items():
        if ik<=n:
            del freq1[it]
    for it,ik in dict(freq2).items():
        if ik<=n:
            del freq2[it]
    for it,ik in dict(freq3).items():
        if ik<=n:
            del freq3[it]
    for it,ik in dict(freq4).items():
        if ik<=n:
            del freq4[it]
    for it,ik in dict(freq).items():
        if ik<=n:
            del freq[it]
    
    wordlist=[w for w in wordlist if w.count1+w.count2+w.count3+w.count4>n ]

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
    i=0
    for w in wordlist:
        i+=1
        f.write(str(i)+'  '+w.disp()+'\n')  
    f.close()
    
    return wordlist

def baseline():
    global start
    print("--------------------------------------------------------")     
    print("Baseline model")
    train_data()
    print("--------------------------------------------------------")     
    print("word count:",storycount,askcount,showcount,pollcount)
    print("Vocabulary size:",vocabsize)
    print("--------------------------------------------------------")     
    print("Check file for output..\n")
    test_data(wordlist,"Output\\baseline-result.txt")
    print("\nTotal elapsed time:",round(time.time() - start,1),"seconds")
    print("--------------------------------------------------------")     
#     '''
#     Evaulates again after removing stop words
#     '''
    
    input("Press any key to remove stop words")    
    start= time.time()
    print("--------------------------------------------------------")     
    print("\n\nTrying again with stop words...\n")
    print("--------------------------------------------------------")     
    stoplist=remove_stopwords()
    print("word count:",storycount,askcount,showcount,pollcount)
    print("Vocabulary size:",vocabsize)
    update_freq()
    test_data(stoplist,"Output\\stopword-result.txt")
    print("\nTotal elapsed time:",round(time.time() - start,1),"seconds")
    print("--------------------------------------------------------")     
    
#     '''
#     Evaulates again after filtering length
#     '''     
    input("Press any key to filter length")    
    start= time.time()
    print("--------------------------------------------------------")     
    print("\n\nTrying again with length filter...\n")
    print("--------------------------------------------------------")     
    lis=filterlength()
    print("word count:",storycount,askcount,showcount,pollcount)
    print("Vocabulary size:",vocabsize)
    update_freq()
    test_data(lis,"Output\\wordlength-result.txt")
#     os.system("notepad.exe Output\\model-2018.txt")
#     os.system("notepad.exe Output\\baseline-result.txt")
    end = time.time()
    print("\nTotal elapsed time:",round(end - start,1),"seconds")
    
    
# ''' 
# This performs experiments with the testing by gradually in filtering the word
# frequencies
# ''' 
def gradualfreq():
    global wordlist
    train_data()
    print()
    freqfilter(1)   
    update_freq()
    x=[]
    y=[]
    print("\n-------------------------------------------------------")        
    print("freq filter value:",1)
    val=test_data(wordlist, "Output\\frequency_filter_output\\smoothfilter_1_result.txt")
    x.append(1)
    y.append(val)
    print("Vocabulary size:",vocabsize)
    print("-------------------------------------------------------")        
    createmodelfile("Output\\frequency_filter_output\\gradual_freq_1_model-2018.txt")
    for i in range(1, 5):
        i=i*5
        x.append(i)
        print()
        freqfilter(i)
        update_freq()
        print("\n-------------------------------------------------------")     
        print("Freq filter value:",i)
        val=test_data(wordlist, "Output\\frequency_filter_output\\smoothfilter_"+str(i)+"_result.txt")
        y.append(val)
        createmodelfile("Output\\frequency_filter_output\\gradual_freq_"+str(i)+"_model-2018.txt")
        print("Vocabulary size:",vocabsize)
        print("-------------------------------------------------------")     
    end = time.time()
    print("\nTotal elapsed time:",round(end - start,1),"seconds")    
    plt2.plot(x,y) 
    plt2.ylabel('Performance')
    plt2.xlabel('Frequency')
    plt2.show()
   
# ''' 
# Saves model into text file
# '''     
def createmodelfile(filename):
    global wordlist
    f = open(filename, "w",encoding='ISO-8859-1')
    wordlist = sorted(wordlist, key=lambda x: x.word, reverse=False)
    i=0
    for w in wordlist:
        i+=1
        f.write(str(i)+'  '+w.disp()+'\n')  
    f.close() 

# ''' 
# This performs experiments with the testing by gradually changing the smoothing value
# frequencies
# '''         
def gradualsmooth():
    global wordlist
    train_data()
    x=[]
    y=[]
    for i in range(1, 11):
        i=round(i*0.1,2)
        x.append(i)
        print()
        update_smooth(i)
        print("\n\n--------------------------------------------------------")     
        print("smooth value:",i)
        val=test_data(wordlist, "Output\\smooth_filter_output\\smoothfilter_"+str(i)+"_result_.txt")
        y.append(val)
        createmodelfile("Output\\smooth_filter_output\\gradual_smooth_"+str(i)+"_model-2018.txt")
        print("--------------------------------------------------------")     
    end = time.time()
    print("\nTotal elapsed time:",round(end - start,1),"seconds")
    plt.plot(x, y)  
    plt.ylabel('Performance')
    plt.xlabel('smooth value')
    plt.show()  


# ''' 
# User interface
# '''     
filename="Resources\\sample"+input("Enter input file:")+".csv"    
print(filename)
ch= input('1.Stop word, word filter \n2.Gradual smooth filter\n3.Gradual frequency filter\n Enter option:')
start = time.time()
if(ch=="1"):
    baseline()
if(ch=="2"):
    gradualsmooth()
if(ch=="3"):
    gradualfreq()


