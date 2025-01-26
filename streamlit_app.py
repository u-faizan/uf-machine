import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info('This is an app builds a Machine Learning Model!')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
  df

  st.write('**X**')
  x_raw = df.drop('species',axis=1)
  x_raw

  st.write('**Y**')
  y_raw = df.species
  y_raw

with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

with st.sidebar:
  st.header('Input Feactures')
  # "species","island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
  
  island = st.selectbox(
    "On which island was the penguin found?",
    ('Biscoe', 'Dream', 'Torgersen'),
  )

  gender = st.selectbox(
    "What is the gender of the penguin?",
    ('male','female'),
  )
  
  bill_length_mm = st.slider("What is the length of the penguin's bill in millimeters?", 32.10, 59.6, 43.9)
  bill_depth_mm = st.slider("What is the depth of the penguin's bill in millimeters?", 13.10, 21.50, 17.20)
  flipper_length_mm = st.slider("What is the length of the penguin's flipper in millimeters?", 172.0, 231.0, 201.0)
  body_mass_g = st.slider("What is the body mass of the penguin in grams?", 2700.0, 6300.0, 4207.0)
  
   # Create a DataFrame for the input features
  data = {'island': island,
          'bill_length_mm': bill_length_mm,
          'bill_depth_mm': bill_depth_mm,
          'flipper_length_mm': flipper_length_mm,
          'body_mass_g': body_mass_g,
          'sex': gender}
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, x_raw], axis=0)

#input_df

with st.expander('Input features'):
  st.write('**Input penguin**')
  input_df
  st.write('**Combined penguins data**')
  input_penguins

# Data preparation
# Encode X
encode = ['island', 'sex']
df_penguins = pd.get_dummies(x_raw, prefix=encode)

X = df_penguins[:]
X
input_row = df_penguins[:1]



