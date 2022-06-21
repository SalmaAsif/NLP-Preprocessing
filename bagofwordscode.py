import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

paragraph =  """The environment is the basic life support system for all living things on planet Earth. 
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
               


ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    sentence = re.sub('[^a-zA-Z]', ' ', sentences[i])
    sentence = sentence.lower()
    sentence = sentence.split()
    sentence = [ps.stem(word) for word in sentence if not word in set(stopwords.words('english'))]
    sentence = ' '.join(sentence)
    corpus.append(sentence)
    
# Creating the Bag of Words model

cv = CountVectorizer(max_features = 500)
X = cv.fit_transform(corpus).toarray()
