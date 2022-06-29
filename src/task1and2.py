import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#%matplotlib inline
import matplotlib as mpl
import plotly.graph_objects as go

import plotly.express as px

pd.set_option('display.max_rows',500)


#Data preparation
#Taking the data from the url and reading it into dataframe

url="https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
df=pd.read_csv(url)

df.isnull().sum() #Checking the NaN values column wise

#Data Preparation
#Data for each country is considere3d as a bool and the or operation has been performed for the required countries
df_can=df['location']=='Canada'
df_ind=df['location']=='India'
df_ger=df['location']=='Germany'
df_list= df_can | df_ind |df_ger
df_countries=df[df_list]         

#plotting the cases spread for 3 countries
fig=px.line(df_countries, x= "date", y= "total_cases",color='location', title= "Covid Cases spread")
fig.show()

#Figure 1 The relative cases overtime of Covid infectors 
df_countries['total_cases_per_population']=df_countries['total_cases']/df_countries['population'] #absolut Covid cases/population size
#plotting the relative spread for 3 countries
fig=px.line(df_countries, x= "date", y= "total_cases_per_population",color='location', title= "Covid Cases spread per population")
fig.update_xaxes(range=["2020-08-01","2022-06-25"])
fig.show()

cols=['people_fully_vaccinated']
df_countries.loc[:,cols] = df_countries.loc[:,cols].ffill() #to cover up for missing data

df_countries['people_fully_vaccinated'] = df_countries['people_fully_vaccinated'].fillna(0)

df_countries.reset_index(drop=True)

#Figure 2 The vaccination rate (percentage of the population) over time 
df_countries['vaccination_rate']=df_countries['people_fully_vaccinated']/df_countries['population'] #absolut Covid cases/population size
#plotting the relative vaccination rate for 3 countries
fig=px.line(df_countries, x= "date", y= "vaccination_rate",color='location', title= "Covid vaccination rate per population")
fig.update_xaxes(range=["2021-03-01","2022-06-25"]) 
fig.update_yaxes(range=[0,1])
fig.show()