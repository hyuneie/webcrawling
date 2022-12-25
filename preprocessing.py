#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install xlrd

# In[25]:


from google.colab import files
myfile = files.upload()

# In[2]:


import io
import pandas as pd

# In[26]:


df = pd.read_excel('GS_TOTAL (1).xlsx')

df

# In[ ]:


# # 특수문자 지우기

# df["review"] = df["review"].str.replace(pat=r'[^\w\s]', repl=r' ', regex=True)

# df

# In[ ]:


# df["Tweets"] = df["Tweets"].str.replace(pat=r'[^a-zA-Z가-힣]', repl = r' ', regex=True)

# df

# In[ ]:


#  # 한글 자음, 모음 제거
# df["Tweets"] = df["Tweets"].str.replace(pat=r'([ㄱ-ㅎㅏ-ㅣ])', repl = r' ', regex=True)

# #숫자 제거
# df["Tweets"] = df["Tweets"].str.replace(pat=r'[0-9]', repl = r' ', regex=True)

# df

# In[ ]:


# # RT라는 글자 지우기

# df["Tweets"] = df["Tweets"].str.replace('RT', ' ', regex=True)

# df

# In[ ]:


# # 영어 지우기

# df["Tweets"] = df["Tweets"].str.replace(pat=r'[A-Za-z]', repl=r' ', regex=True)

# df

# In[27]:


df["review"] = df["review"].str.replace("  ", "")

df


# In[ ]:




# In[28]:


# 공백 기준 분할

df2 = df["review"]

df2

# In[29]:


# 글자 2개 이상인 경우만 출력

df_list = [] 
for word in df2:
    if len(word) > 2:
        df_list.append(word)

print(df_list)

# In[ ]:




# In[30]:


# 리스트 다시 데이터프레임화

df_list2 = pd.DataFrame(zip(df_list))

display(df_list2)

# In[31]:


# 엑셀로 저장
df_list2.to_excel('GS_TOTAL2.xlsx')
