# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 11:26:03 2021

@author: salma asif
"""

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

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
               
               
sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()

# Stemming
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)   
print(sentences)