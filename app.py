import streamlit as st 
from transformers import pipeline

pipe = pipeline("text-classification", model="djangodevloper/bert-base-uncased-product-review-classification")
st.title('Product Review Classification')
text = st.text_area('Enter your comment ... ')
review = st.button('Rewiew ' , type='primary')
if review:
    if text:
        result = pipe(text)
        st.info(f"{result[0]['label']} : {result[0]['score']}")
    else:
        st.info("Kindly enter your review")