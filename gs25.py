#!/usr/bin/env python
# coding: utf-8

# In[2]:


df = pd.read_csv('naver_gs25.csv')

# In[3]:


df["내용"] = df["내용"].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df

# In[4]:


df=df.drop(['Unnamed: 0'], axis = 1)

# In[5]:


df=df.drop(['링크'], axis = 1)

# In[6]:


context = df['내용'].tolist()
print(len(context))

context=' '.join(context)
context=context[:]
print(context)

# In[ ]:


#articles = articles.replace('(', '')

# In[7]:


from konlpy.tag import Kkma
kkma = Kkma()	# 아마 설치가 잘 되지 않았다면 이 단계에서 에러가 났을 것이다.
print(kkma.nouns(u'안녕하세요 Soo입니다'))

# In[8]:


from konlpy.tag import Okt
okt = Okt()

# In[9]:


tokenizer = Okt()
raw_pos_tagged = tokenizer.pos(context, norm=True, stem=True) # POS Tagging
print(raw_pos_tagged)

# In[44]:


del_list = ['를', '이', '은', '는', '있다', '하다', '에', '있다', '되다', '이다', '돼다', '않다', '그렇다', '아니다', '이렇다', '그렇다', '어떻다', '없다', '같다','보다','GS','지에스','gs',]

word_cleaned = []
for word in raw_pos_tagged:
    if not word[1] in ["Josa", "Eomi", "Punctuation", "Foreign", "KoreanParticle", "Number"]: # Foreign == ”, “ 와 같이 제외되어야할 항목들
        if (len(word[0]) != 1) & (word[0] not in del_list): # 한 글자로 이뤄진 단어들을 제외 & 원치 않는 단어들을 제외, 대신 "안, 못"같은 것까지 같이 지워져서 긍정,부정을 파악해야 되는경우는 제외하지 않는다.
            word_cleaned.append(word[0])
        
print(word_cleaned)

# In[45]:


from collections import Counter

# In[46]:


result = Counter(word_cleaned)
word_dic = dict(result)
print(word_dic)

# In[47]:


sorted_word_dic = sorted(word_dic.items(), key=lambda x:x[1], reverse=True)
print(sorted_word_dic)

# In[48]:


import nltk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 그래프에 한글 폰트 설정
font_name = matplotlib.font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name() # NanumGothic.otf
matplotlib.rc('font', family=font_name)


word_counted = nltk.Text(word_cleaned)
plt.figure(figsize=(15,7))
plt.title('네이버 검색어: -gs25-글 에서 쓰인 단어 빈도수 ')

word_counted.plot(50)

# In[49]:


word_frequency = nltk.FreqDist(word_cleaned)
df = pd.DataFrame(list(word_frequency.values()), word_frequency.keys())

result = df.sort_values([0], ascending = False)
result = result[:50]
result.plot(kind='bar', legend=False, figsize=(15,5))
plt.title('네이버 검색어: -gs25-글 에서 쓰인 단어 빈도수')
plt.show()

# In[50]:


# 네이버 검색어 : gs25

# In[43]:


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

# In[3]:


jupyter nbconvert --to script gs25__양은지.ipynb
