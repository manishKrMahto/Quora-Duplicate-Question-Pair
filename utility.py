import numpy as np
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
from fuzzywuzzy import fuzz
import nltk 
nltk.download('punkt_tab')

stop_words = stopwords.words('english')

cv = pickle.load(open('cv.pkl' , 'rb'))
rf = pickle.load(open('rf.pkl' , 'rb'))
scaler = pickle.load(open('scaler.pkl' ,'rb'))


def preprocess(q1 , q2):
    lst = []
    # expending like i'll --> i will
    q1 = contractions.fix(q1)
    q2 = contractions.fix(q2)

    # lowercasing and strip
    q1 = q1.lower().strip()
    q2 = q2.lower().strip()
    
    # CUSTOM features 
    q1_len = len(q1)
    q2_len = len(q2)
    
    q1_words = len(q1.split(" "))
    q2_words = len(q2.split(" "))
    
    word1 = set(q1.split(" "))
    word2 = set(q2.split(" "))

    word_common = len(word1.intersection(word2))
    
    word_total = q1_len + q2_len
    
    word_share = word_common / word_total
        
    q1_stopwords = len(find_word_tokenize(q1))
    q2_stopwords = len(find_word_tokenize(q2))
    
    cwc_min = word_common / (min(q1_words , q2_words) + 0.000001)
    cwc_max = word_common / (max(q1_words , q2_words) + 0.000001)
    
    csc_min = word_common/ (min(q1_stopwords , q2_stopwords) + 0.000001)
    csc_max = word_common/ (max(q1_stopwords , q2_stopwords) + 0.000001)
    
    ctc_min = word_common / (min(q1_words , q2_words) + 0.0000001)
    ctc_max = word_common / (max(q1_words , q2_words) + 0.0000001)
    
    first_word_eq = 1 if q1.split(" ")[0] == q2.split(" ")[0] else 0
    last_word_eq = 1 if q1.split(" ")[-1] == q2.split(" ")[-1] else 0
    
    # LENGTH based features 
    
    mean_len = (q1_len + q2_len) / 2
    abs_len_diff = abs(q1_words - q2_words)
    
    
    fuzzy_features = fetch_fuzzy_features(q1 , q2)
    
    fuzz_ratio = fuzzy_features[0]
    fuzz_partial_ratio = fuzzy_features[1]
    token_sort_ratio = fuzzy_features[2]
    token_set_ratio = fuzzy_features[3]
    
    scaled_values = scaler.transform([[cwc_min, cwc_max, csc_min, csc_max , ctc_min , ctc_max , last_word_eq, first_word_eq , abs_len_diff , mean_len , token_set_ratio , token_sort_ratio ,  fuzz_ratio , fuzz_partial_ratio]])
    
    
    q1_bow = cv.transform([q1]).toarray()
    q2_bow = cv.transform([q2]).toarray()
    
    basic_features = np.array([q1_len, q2_len, q1_words, q2_words, word_common , word_total, word_share, q1_stopwords,q2_stopwords])
    
    # print(basic_features.shape)
    # print(scaled_values.shape)
    # print(q1_bow.shape)
    # print(q2_bow.shape)
    # print(np.hstack((basic_features.reshape(1,-1) , scaled_values.reshape(1,-1) , q1_bow.reshape(1,-1) , q2_bow.reshape(1,-1))).shape)
    
    input_array = np.hstack((basic_features.reshape(1,-1) , scaled_values.reshape(1,-1) , q1_bow.reshape(1,-1) , q2_bow.reshape(1,-1)))

    # print(rf.predict(input_array))
    
    return input_array
    



def find_word_tokenize(text):
    stop_words = stopwords.words('english')

    words = word_tokenize(text)

    return ([word for word in words if word in stop_words])


def fetch_fuzzy_features(q1 ,q2):

    # if error happens
    fuzzy_features = [0.0]*4

    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features


    
if __name__ == '__main__':
    # not duplicate --> 0
    # preprocess("What is the step by step guide to invest in share market in india?" , "What is the step by step guide to invest in share market?")

    # duplicate --> 1 
    preprocess("Do you believe there is life after death?" , "Is it true that there is life after death?")