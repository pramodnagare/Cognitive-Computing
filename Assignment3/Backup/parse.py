# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 17:21:49 2018

@author: wenqi
"""
import json
def readdata(file_name):
    f = open(file_name, mode='r')
    return f.read()
amazon_q3 = readdata('Amazon.com (AMZN) Q3 2018 Results - Earnings Call Transcript.txt')
list_q3 = amazon_q3.split('\n')
i = 0
while i < len(list_q3):
    if list_q3[i] == '':
        del list_q3[i]
    else:
        i+=1
n = len(list_q3)
dict_q3 = dict(zip(range(1,n+1),list_q3))
dict_sentiment = dict(zip(range(1,n+1),['neutral']*n))
dict_fin = {'text':dict_q3,'sentiment':dict_sentiment}
with open('amazon_data.json', 'w') as f:
    json.dump(dict_fin, f)
    
with open('amazon_data_f.json','r') as f:
    dict_fin_f = json.load(f)

div_len = 51;
list_q3_s=[]
list_q3_l=[]
for i in range(len(list_q3)):
    if len(list_q3[i])<div_len :
        list_q3_s.append(list_q3[i])
    else:
        list_q3_l.append(list_q3[i])

dict_q3_l = dict(zip(range(1,len(list_q3_l)+1),list_q3_l))
dict_sentiment_l = dict(zip(range(1,len(list_q3_l)+1),['neutral']*len(list_q3_l)))
dict_fin = {'text':dict_q3_l,'sentiment':dict_sentiment_l,'exclude':list_q3_s}
with open('amazon_data_ex.json', 'w') as f:
    json.dump(dict_fin, f)

    
