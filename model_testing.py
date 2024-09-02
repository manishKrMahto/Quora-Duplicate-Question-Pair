import streamlit as st
from utility import preprocess  
from calculate_similarity_score import  find_similarity_score
import pickle
import pandas as pd

rf = pickle.load(open('rf.pkl' , 'rb'))

st.title('Quara Duplicate Question Pair Model Testing')

df = pd.read_csv('quara duplicate questions pair.csv')

sample_size = 0
sample_size = st.text_input('Enter Sample size : ')
st.write(f'sample size : {sample_size}')

status = st.button('Do Testing')

if status :
    new_df = df[['question1' ,'question2','is_duplicate']].sample(int(sample_size) , random_state=2)

    for i in range(new_df.shape[0]):
        q1 = new_df['question1'].iloc[i]
        q2 = new_df['question2'].iloc[i]
        
        
        input_array = preprocess(q1,q2)
        sim_score = round(float(find_similarity_score(q1 , q2)) , 2)
        if rf.predict(input_array)[0] == 1:
            st.success(f'Duplicate Question Pair with {sim_score}% of probability')
        else :
            st.warning(f'Not Duplicate Question Pair with {sim_score}% of probability')
            
        st.write(new_df.iloc[i])