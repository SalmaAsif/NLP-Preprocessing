import nltk

from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re

paragraph = """The environment is the basic life support system for all living things on planet Earth. 
                It is a combination of natural and human-made components. Natural components include air, 
                water, land and living organisms. Roads, industries, buildings, etc., are human-made components.
                The natural environment can be differentiated into four main components – Biosphere, Lithosphere,
                Hydrosphere and Atmosphere. The topmost layer of the Earth is called the Lithosphere, which is a 
                thin layer of soil made of rocks and minerals. The hydrosphere consists of various types of water
                bodies like seas, oceans, rivers, lakes, ponds, etc. Atmosphere, consisting of water vapour, gases
                and dust particles, is the layer of air that surrounds the Earth. The living world consisting of 
                human beings, plants and animals constitute the biosphere.
                The environment is dependent on the interaction between all the different components. However,
                human beings play a huge role in the making and breaking of the environment. Being the supreme 
                most intellectual power on Earth, human beings influence the wellness of the environment to a 
                great extent. The impact of the environment on all living beings is directly proportional to the 
                way human beings treat the environment. Any kind of existence would not be possible without air,
                water or land. Nothing to eat, not a drop to drink and nowhere to go is not what we or our future 
                generations should expect to have. Every living thing depends largely on the environment for
                survival, and having a clean and safe environment is solely in the hands of the human beings."""



# Preprocessing the data
#text = re.sub(r'\[[0-9]*\]',' ',paragraph)
#text = re.sub(r'\s+',' ',text)
text = paragraph.lower()
#text = re.sub(r'\d',' ',text)
#text = re.sub(r'\s+',' ',text)

# Preparing the dataset
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    
    
# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)

# Finding Word Vectors
vector = model.wv['environment']
print(vector)

# Most similar words
similar = model.wv.most_similar('human')
print(similar)
