#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from konlpy.tag import Mecab 

# In[2]:


# pip install --upgrade pandas

# In[3]:


# pip install --upgrade pip 

# In[4]:


# pip install konlpy 

# In[5]:


# pip install mecab_python-0.996_ko_0.9.2_msvc-cp38-cp38-win_amd64.whl

# In[6]:


#pip install --upgrade pip

# In[7]:


# pip install JPype1-1.4.0-cp39-cp39-win_amd64.whl

# In[8]:


pip install mecab_python-0.996_ko_0.9.2_msvc-cp38-cp38-win_amd64.whl

# In[9]:


from konlpy.tag import Komoran 
komoran = Komoran() 
text = "아버지가 방에 들어가신다." 
komoran.nouns(text)
komoran.morphs(text)

# In[40]:


import MeCab

m = MeCab.Tagger()
out = m.parse("아빠가가방에들어가신다.")

print(out)

# In[41]:


import MeCab

m = MeCab.Tagger()
out = m.parse("파이썬 에서 형태소 분석하기.")

print(out)

# In[36]:


import re

# Basic Cleaning Text Function
def CleanText(readData, Num=False, Eng=False):

    # Remove Retweets 
    text = re.sub('RT @[\w_]+: ', '', readData)

    # Remove Mentions
    text = re.sub('@[\w_]+', '', text)

    # Remove or Replace URL 
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ' ', text) # http로 시작되는 url
    text = re.sub(r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{2,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", ' ', text) # http로 시작되지 않는 url
    
    # Remove Hashtag
    text = re.sub('[#]+[0-9a-zA-Z_]+', ' ', text)

    # Remove Garbage Words (ex. &lt, &gt, etc)
    text = re.sub('[&]+[a-z]+', ' ', text)

    # Remove Special Characters
    text = re.sub('[^0-9a-zA-Zㄱ-ㅎ가-힣]', ' ', text)
    
    # Remove newline
    text = text.replace('\n',' ')
    
    if Num is True:
        # Remove Numbers
        text = re.sub(r'\d+',' ',text)
    
    if Eng is True:
        # Remove English 
        text = re.sub('[a-zA-Z]' , ' ', text)

    # Remove multi spacing & Reform sentence
    text = ' '.join(text.split())
       
    return text

# In[37]:


SAMPLE_TEXT = "<🚨코로나19 가짜뉴스 팩트체크>\n\n신천지가 기성교회에 가서 코로나를 전파하라고 했다??!\n\n- 사실 무근입니다. \n신천지는 2월 18일 부터 전국 교회를 폐쇄하고 온라인\n예배로 전환하였습니다.\n\n#온라인예배\n#가짜뉴스_이제그만\n#신천지_팩트체크 pic.twitter.com/Dppie6iean"

print(f"Before cleaning text:\n{SAMPLE_TEXT}")
print("\n")
print(f"After cleaning text:\n{CleanText(SAMPLE_TEXT)}")
print("\n")
print(f"After cleaning text when Num is True:\n{CleanText(SAMPLE_TEXT, Num=True)}")

# In[34]:


from konlpy.tag import Mecab 

# Preprocessing code with Mecab
mecab = Mecab(dicpath="C:\mecab\mecab-ko-dic") # Mecab User Dic Path

def preprocessing_mecab(readData):
    
    #### Clean text
    sentence = CleanText(readData)
    
    #### Tokenize
    morphs = mecab.pos(sentence)
    
    JOSA = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"] # 조사
    SIGN = ["SF", "SE", "SSO", "SSC", "SC", "SY"] # 문장 부호
    TERMINATION = ["EP", "EF", "EC", "ETN", "ETM"] # 어미
    SUPPORT_VERB = ["VX"] # 보조 용언
    NUMBER = ["SN"]
    
    # Remove JOSA, EOMI, etc
    morphs[:] = (morph for morph in morphs if morph[1] not in JOSA+SIGN+TERMINATION+SUPPORT_VERB)
        
    # Remove length-1 words  
    morphs[:] = (morph for morph in morphs if not (len(morph[0]) == 1))
    
    # Remove Numbers
    morphs[:] = (morph for morph in morphs if morph[1] not in NUMBER)
   
    # Result pop-up
    result = []
    for morph in morphs:
        result.append(morph[0])
        
    return result

# In[38]:


SAMPLE_TEXT = "<🚨코로나19 가짜뉴스 팩트체크>\n\n신천지가 기성교회에 가서 코로나를 전파하라고 했다??!\n\n- 사실 무근입니다. \n신천지는 2월 18일 부터 전국 교회를 폐쇄하고 온라인\n예배로 전환하였습니다.\n\n#온라인예배\n#가짜뉴스_이제그만\n#신천지_팩트체크 pic.twitter.com/Dppie6iean"

# RUN!
preprocessing_mecab(SAMPLE_TEXT)

print(out)

# In[43]:


from konlpy.tag import Kkma
kkma = Kkma()	# 아마 설치가 잘 되지 않았다면 이 단계에서 에러가 났을 것이다.
print(kkma.nouns(u'안녕하세요 Soo입니다'))

# In[44]:


from konlpy.tag import Okt
okt = Okt()

# In[45]:


#전처리과정
df = pd.read_csv('gs편의점_2022(대cvs)_1.csv')
df.head()

# In[46]:


df["Context"] = df["Context"].str.replace(pat=r'[^\w]', repl=r'', regex=True)
#특수문자제거

# In[47]:


df["Context"] = df["Context"].str.replace(pat=r'["_"]', repl=r'', regex=True)
#_ 제거

# In[50]:


df["Context"] = df["Context"].str.replace(pat=r'["^a-zA-Z0-9"]', repl=r'', regex=True)
# 영어,숫자제거

# In[51]:


df.head()

# In[53]:


# 한줄로 묶기
context = df['Context'].tolist()
print(len(context))

context=' '.join(context)
context=context[:]
print(context)

# In[54]:


!pip install matplotlib-venn

# In[55]:


from konlpy.tag import Okt
okt = Okt()

# In[56]:


tokenizer = Okt()
raw_pos_tagged = tokenizer.pos(context, norm=True, stem=True) # POS Tagging
print(raw_pos_tagged)

# In[57]:


#불용어 처리 리스트

# In[58]:


del_list = ['를', '이', '은', '는', '있다', '하다', '에']  

word_cleaned = []
for word in raw_pos_tagged:
    if not word[1] in ["Josa", "Eomi", "Punctuation", "Foreign"]: # Foreign == ”, “ 와 같이 제외되어야할 항목들
        if (len(word[0]) != 1) & (word[0] not in del_list): # 한 글자로 이뤄진 단어들을 제외 & 원치 않는 단어들을 제외, 대신 "안, 못"같은 것까지 같이 지워져서 긍정,부정을 파악해야 되는경우는 제외하지 않는다.
            word_cleaned.append(word[0])
        
print(word_cleaned)

# In[59]:


from collections import Counter

# In[60]:


result = Counter(word_cleaned)
word_dic = dict(result)
print(word_dic)

# In[61]:


#카운터 기본 사용법
#https://www.daleseo.com/python-collections-counter/

# In[62]:


# # 그후 sorted( ) 함수로 정렬해줍니다. key파라미터에 단어의 개수를 전달인자로 하는 lambda함수를 통해 정렬해줍니다.

# word_dic.imtes( )를 통해 딕셔너리의 key, value쌍을 튜플로 받습니다.
# 이 튜플에서 lambda함수에 output값으로 x[1]을 설정하여 단어의 개수를 key파라미터의 값으로 합니다.

# In[63]:


sorted_word_dic = sorted(word_dic.items(), key=lambda x:x[1], reverse=True)
print(sorted_word_dic)

# In[67]:


import nltk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 그래프에 한글 폰트 설정
font_name = matplotlib.font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name() # NanumGothic.otf
matplotlib.rc('font', family=font_name)


word_counted = nltk.Text(word_cleaned)
plt.figure(figsize=(15,7))
plt.title('트위터 검색어: -GS편의점-글 에서 쓰인 단어 빈도수 ')

word_counted.plot(50)



# In[66]:


# pip install nltk

# In[68]:


word_frequency = nltk.FreqDist(word_cleaned)
df = pd.DataFrame(list(word_frequency.values()), word_frequency.keys())

result = df.sort_values([0], ascending = False)
result = result[:50]
result.plot(kind='bar', legend=False, figsize=(15,5))
plt.title('트위터 검색어: -GS편의점-글 에서 쓰인 단어 빈도수')
plt.show()

# In[ ]:




# In[69]:


from wordcloud import WordCloud 
import matplotlib.pyplot as plt
import numpy as np
from PIL import *

# In[14]:


# pip install wordcloud

# In[70]:


cand_mask=np.array(Image.open('circle.jpg'))

# In[80]:


cand_mask=np.array(Image.open('circle.jpg'))

# In[94]:


sorted_word_dic

# In[99]:


#배열값 메모장에 넣기

# In[103]:


# pip install numpy

# In[104]:


import numpy as np

# In[111]:


f = open("sorted_word_dic.txt",'w')
f.write(str(sorted_word_dic))
f.close()

# In[112]:


from operator import itemgetter

# In[116]:


print(sorted_word_dic[:5])

# In[136]:


word_dict ={}
for n, i in sorted_word_dic[:10]:
    word_dict[n]=i
    
    print("편의점 wordcount:",word_dict['편의점'])
    print("딕셔너리갯수:",len(word_dict))

# In[134]:


wc = WordCloud(font_path = "malgun",
        max_words = 100,
        max_font_size = 200)
cloud = wc.fit_words(word_dict)
cloud.to_image() # 상위 20개 단어


# In[137]:


wc = WordCloud(font_path = "malgun",
        max_words = 100,
        max_font_size = 200)
cloud = wc.fit_words(word_dict)
cloud.to_image() #상위 10 단어


# In[ ]:



