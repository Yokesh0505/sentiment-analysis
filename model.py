#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
import pickle
nltk.download('vader_lexicon')
text = 'This is a very nice day'

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
score = ((sid.polarity_scores(str(text))))['compound']


pickle.dump((score),open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


# In[ ]:




