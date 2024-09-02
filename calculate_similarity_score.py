import pickle
from gensim.utils import simple_preprocess
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk 
nltk.download('stopwords')

Word2Vec_model = pickle.load(open('Word2Vec_model.pkl' , 'rb'))


def find_word_2_vec(Word2Vec_model , question):

  word_vectors = [Word2Vec_model.wv[word] for word in simple_preprocess(question) if word in Word2Vec_model.wv]
  # print(word_vectors)

  if len(word_vectors) > 0 :
    # making all word vectors to MEAN of all word vectors for that sentence
    return np.mean(word_vectors , axis = 0)
  else :
    # Handle the case where no word in the sentence exists in the model's vocabulary
    return np.zeros(Word2Vec_model.vector_size)





def remove_stopwords(text):
  new_text = []

  for word in text.split():
    if word in stopwords.words('english'):
      new_text.append('')
    else:
      new_text.append(word)

  x = new_text[:]
  new_text.clear()

  return " ".join(x)


def find_similarity_score(q1 ,q2):
  
  q1 = remove_stopwords(q1)
  q2 = remove_stopwords(q2)

  # print(q1)
  # print(q2)

  vec1 = find_word_2_vec(Word2Vec_model , q1)
  vec2 = find_word_2_vec(Word2Vec_model , q2)

  # print(vec1.shape)
  # print(vec2.shape)

  sim = cosine_similarity(vec1.reshape(1,-1),vec2.reshape(1,-1))

  return sim[0][0]
  
  
if __name__ == '__main__':
  # q1 , q2 = "Do you believe there is life after death?" , "Is it true that there is life after death?"

  q1 , q2 = "What is the step by step guide to invest in share market in india?" , "What is the step by step guide to invest in share market?"

  print(find_similarity_score(q1,q2))