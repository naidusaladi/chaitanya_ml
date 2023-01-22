# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 11:51:28 2023

@author: DELL
"""

import numpy as np

import pickle

import streamlit as st

ra_model=pickle.load(open("https://github.com//naidusaladi//chaitanya_ml//blob//main//ra_model.sav",'rb'))

mrr_model=pickle.load(open("https://github.com//naidusaladi//chaitanya_ml//blob//main//mrr_model.sav",'rb'))

# creating a function for prediction

def prediction(input_data):
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    
    ra_predict=ra_model.predict(input_data_reshaped)
    
    mrr_predict=mrr_model.predict(input_data_reshaped)
    
    return [ra_predict , mrr_predict]


def main():
    #giving a title 
    st.title("Project Name")
    #input data from user
    s=st.text_input("Enter 'S' value :")
    f=st.text_input("Enter 'F' value :")
    d=st.text_input("Enter 'D' value :")


    
    #code for prediction
    output=[0,0,0]
    
    #creating a button for prediction 
    
    if st.button("Predict"):
        if(s!="" and f!="" and d!=""):
            st.write(type(s))
            output=prediction([s,f,d])
            st.write("Predicted Output :")
            st.write("ra : "+str(output[0][0]))
            st.write(" mrr : "+str(output[1][0]))
        else:
            st.write("Enter data Correct data")
    
if __name__=='__main__' :
    main()
    
        
