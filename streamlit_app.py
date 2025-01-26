import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info('This is an app builds a Machine Learning Model!')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
  df

  st.write('**X**')
  x = df.drop('species',axis=1)
  x

  st.write('**Y**')
  y = df.species
  y

with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

with st.sidebar:
  st.header('Input Feactures')
  # "species","island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
  island = st.selectbox(
    "On which island was the penguin found?",
    ('Biscoe', 'Dream', 'Torgersen'),
  )
  



