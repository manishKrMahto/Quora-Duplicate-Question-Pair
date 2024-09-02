import streamlit as st
from utility import preprocess  
from calculate_similarity_score import  find_similarity_score
import pickle

rf = pickle.load(open('rf.pkl' , 'rb'))

st.title('Quara Duplicate Question Pair')

q1 = st.text_input('Enter question 1 ')
q2 = st.text_input('Enter question 2 ')


st.write(f'Question 1 : {q1}')
st.write(f'Question 2 : {q2}')

status = st.button('What ?')

if status :
    input_array = preprocess(q1,q2)
    sim_score = round(float(find_similarity_score(q1 , q2)) , 2)
    if rf.predict(input_array)[0] == 1:
        st.success(f'Duplicate Question Pair with {sim_score}% of similarity score')
    else :
        st.warning(f'Not Duplicate Question Pair with {sim_score}% of similarity score')



# q1 , q2 = "What is the step by step guide to invest in share market in india?" , "What is the step by step guide to invest in share market?"
