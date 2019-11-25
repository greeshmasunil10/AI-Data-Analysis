import pandas as pd
import os


inputfile="Resources\\sample100.csv"
def trainModel():
    df= pd.read_csv(inputfile,encoding='ISO-8859-1',usecols=['Title', 'Post Type','Created At'])
    f1 = open("op.txt", "w",encoding='ISO-8859-1')
    df = df[(df['Created At']<'2019')]
    print(df.keys())
    print(df[df['Post Type']=="ask_hn"]['Title'] )
    f1.write(str(df))
    f1.close()
    os.system("Notepad.exe op.txt")
#     print(df)
#     print(df.'Post_type')
#     df['PostType'] = df.PostType.map({'story': 0, 'ask_hn': 1, 'show_hn':2, 'post':3})
#     print(df.PostType)    
    
def startprogram():
    trainModel()    

startprogram()    