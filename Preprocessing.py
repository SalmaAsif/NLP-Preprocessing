
## NLP basic preprocessing steps

##import required libraries 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 

## 1. Tokenization: Splitting the sentence into words.

sentence = "Mary and Samantha arrived at the bus station early but waited until noon for the bus."

words = word_tokenize(sentence)
print(words,'\n')


## 2. Lower casing: Converting a word to lower case

sentence = sentence.lower()
print(sentence,'\n')

## 3. Stop words removal:removing the words that occur commonly across all the documents in the corpus.
## Typically, articles and pronouns are generally classified as stop words.

stop_words = set(stopwords.words('english')) 
word_tokens = word_tokenize(sentence)
  
filtered_sentence = [w for w in word_tokens if not w in stop_words] 
print(filtered_sentence,'\n')

## 4. Stemming: It is a process of transforming a word to its root form.

ps = PorterStemmer()

for word in sentence.split():
 print(ps.stem(word))
 
## 5. Lemmatization: lemmatization reduces the words to a word existing in the language.
##Lemmatization is preferred over Stemming because lemmatization does a morphological analysis of the words.

lemmatizer = WordNetLemmatizer()
print('\n')
for word in sentence.split():
    print(lemmatizer.lemmatize(word))
