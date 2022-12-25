#!/usr/bin/env python
# coding: utf-8

# In[7]:


pip install wordcloud

# In[8]:


import numpy as np
import pandas as pd

import nltk
from konlpy.tag import Okt

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

from wordcloud import WordCloud
from PIL import Image
from wordcloud import ImageColorGenerator


# In[ ]:


import csv
 
f = open('max_token.csv', 'r', encoding='CP949')
rdr = csv.reader(f)
for line in rdr:
    print(line)
f.close()    

# In[13]:


word_cleaned=line

# In[14]:


word_dic = {}

for word in word_cleaned:
    if word not in word_dic: # 처음 등장할 시
        word_dic[word] = 1
    else: # 추가 등장시 count ++
        word_dic[word] += 1


# In[ ]:


word_dic

# In[21]:


word_dic.items()

# In[22]:


del word_dic['씨유']

# In[24]:


from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

# In[25]:


sorted_word_dic = sorted(word_dic.items(), key = lambda x:x[1], reverse = True)


# In[26]:


word_counted = nltk.Text(word_cleaned)
plt.figure(figsize=(15, 7))
word_counted.plot(50)


# In[27]:


word_frequency = nltk.FreqDist(word_cleaned) # 단어 빈도 계산

# {단어(key): 수(value)} -> 데이터프레임
df = pd.DataFrame(list(word_frequency.values()), word_frequency.keys()) 

# 빈도 내림차순 정렬
result = df.sort_values([0], ascending=False)

# matplot 그래프
result.plot(kind = 'bar', legend = False, figsize = (25,10), rot=0, fontsize=14) # 'bar' graph
plt.show()


# In[28]:


word_cloud = WordCloud(font_path="C:/Windows/Fonts/malgun.ttf", # 폰트설정
                       width=2000, height=1000, # 워드클라우드 크기(해상도)
                       prefer_horizontal= 0.8, # 가로방향 단어 비율 (0~1)
                       background_color='white',
                       colormap = 'Set2')
                       
word_cloud.generate_from_frequencies(word_dic)

plt.figure(figsize=(15,15))
plt.imshow(word_cloud) # image show
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[29]:


word_cloud.to_file('word_cloud_heart.png')


# In[ ]:




# In[ ]:




# In[ ]:



