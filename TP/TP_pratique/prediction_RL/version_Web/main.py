import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write('''
         # GRP2
         Est une application permettant de predire l'octroi d'un crédit à un client selon son profil
         ''')

#Collecter le profil d'entrée
st.sidebar.header("Le profil du client")

def profil_client_entree():
    Gender=st.sidebar.selectbox('Sexe',('Male','Female'))
    Married=st.sidebar.selectbox('Marié',('Yes','No'))
    Dependents=st.sidebar.selectbox('Enfants',('0','1','2','3+'))
    Education=st.sidebar.selectbox('Education',('Graduate','Not Graduate'))
    Self_Employed=st.sidebar.selectbox('Salarié ou Entrepreneur',('Yes','No'))
    ApplicantIncome=st.sidebar.slider('Salaire du client',150,4000,200)
    CoapplicantIncome=st.sidebar.slider('Salaire du conjoint',0,40000,2000)
    LoanAmount=st.sidebar.slider('Montant du crédit en milier de dollar',9.0,700.0,200.0)
    Loan_Amount_Term=st.sidebar.selectbox('Durée du crédit',(360.0,120.0,240.0,180.0,60.0,300.0,36.0,84.0,12.0))
    Credit_History=st.sidebar.selectbox('Credit_History',(1.0,0.0))
    Property_Area=st.sidebar.selectbox('Property_Area',('Urban','Rural','Semiurban'))
    
    data={
    'Gender':Gender,
    'Married':Married,
    'Dependents':Dependents,
    'Education':Education,
    'Self_Employed':Self_Employed,
    'ApplicantIncome':ApplicantIncome,
    'CoapplicantIncome':CoapplicantIncome,
    'LoanAmount':LoanAmount,
    'Loan_Amount_Term':Loan_Amount_Term,
    'Credit_History':Credit_History,
    'Property_Area':Property_Area
    }
    profil_client=pd.DataFrame(data,index=[0])
    return profil_client

input_df=profil_client_entree()

#Transformer les données d'entrée en données adaptées à notre modèle
#importer la base de données
df=pd.read_csv('train.csv')
credit_input=df.drop(columns=['Loan_ID','Loan_Status'])
entree=pd.concat([input_df,credit_input],axis=0)

# encodage des données
var_cat=['Gender', 'Married', 'Dependents', 'Education','Self_Employed','Credit_History', 'Property_Area']
for col in var_cat:
    dummy=pd.get_dummies(entree[col],drop_first=True)
    entree=pd.concat([dummy,entree],axis=1)
    del entree[col]
#prendre uniquement la premiere ligne
entree=entree[:1]
#afficher les données transformées
st.subheader('Les caracteristiques transformés')
st.write(entree)


#importer le modèle
load_model=pickle.load(open('prevision_credit.pkl','rb'))


#appliquer le modèle sur le profil d'entrée
prevision=load_model.predict(entree)

st.subheader('Résultat de la prévision')
st.write(prevision)
#st.bar_chart(entree)