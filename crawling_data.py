#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


DFgs_n = pd.read_csv('gs편의점_2022(대cvs)_1.csv')

DFgs_n.head()

# In[3]:


DFgs_n["Context"] = DFgs_n["Context"].str.replace(pat=r'[^\w]', repl=r'', regex=True)

# In[4]:


DFgs_n.head()

# In[5]:


DFgs_n["Context"] = DFgs_n["Context"].str.replace(pat=r'["_"]', repl=r'', regex=True)

# In[6]:


DFgs_n.head()

# In[7]:


context = DFgs_n['Context'].tolist()
print(len(context))

context=' '.join(context)
context=context[:]
print(context)

# In[8]:


!pip install matplotlib-venn

# In[9]:


pip install JPype1-1.4.0-cp39-cp39-win_amd64.whl

# In[11]:


pip install konlpy

# In[15]:


from konlpy.tag import Kkma
kkma = Kkma()	# 아마 설치가 잘 되지 않았다면 이 단계에서 에러가 났을 것이다.
print(kkma.nouns(u'안녕하세요 Soo입니다'))

# In[16]:


from konlpy.tag import Okt
okt = Okt()

# In[18]:


tokenizer = Okt()
raw_pos_tagged = tokenizer.pos(context, norm=True, stem=True) # POS Tagging
print(raw_pos_tagged)

# In[ ]:


한글자,단어 불용어들 제거.

# In[19]:


del_list = ['를', '이', '은', '는', '있다', '하다', '에']  

word_cleaned = []
for word in raw_pos_tagged:
    if not word[1] in ["Josa", "Eomi", "Punctuation", "Foreign"]: # Foreign == ”, “ 와 같이 제외되어야할 항목들
        if (len(word[0]) != 1) & (word[0] not in del_list): # 한 글자로 이뤄진 단어들을 제외 & 원치 않는 단어들을 제외, 대신 "안, 못"같은 것까지 같이 지워져서 긍정,부정을 파악해야 되는경우는 제외하지 않는다.
            word_cleaned.append(word[0])
        
print(word_cleaned)

# In[21]:


from collections import Counter

# In[22]:


result = Counter(word_cleaned)
word_dic = dict(result)
print(word_dic)

# In[ ]:


#카운터 기본 사용법
#https://www.daleseo.com/python-collections-counter/

# In[ ]:


# # 그후 sorted( ) 함수로 정렬해줍니다. key파라미터에 단어의 개수를 전달인자로 하는 lambda함수를 통해 정렬해줍니다.

# word_dic.imtes( )를 통해 딕셔너리의 key, value쌍을 튜플로 받습니다.
# 이 튜플에서 lambda함수에 output값으로 x[1]을 설정하여 단어의 개수를 key파라미터의 값으로 합니다.

# In[23]:


sorted_word_dic = sorted(word_dic.items(), key=lambda x:x[1], reverse=True)
print(sorted_word_dic)

# In[ ]:


#  nltk 참고 https://jeonjoon.tistory.com/29

# In[30]:


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



# In[31]:


word_frequency = nltk.FreqDist(word_cleaned)
df = pd.DataFrame(list(word_frequency.values()), word_frequency.keys())

result = df.sort_values([0], ascending = False)
result = result[:50]
result.plot(kind='bar', legend=False, figsize=(15,5))
plt.title('트위터 검색어: -GS편의점-글 에서 쓰인 단어 빈도수')
plt.show()

# In[ ]:


# 참고사이트 https://jeonjoon.tistory.com/m/32

# In[10]:


# import sys

# print(sys.version) 버전확인

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



