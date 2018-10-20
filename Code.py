# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas import Series,DataFrame
import matplotlib.patches as mpatches
"""import os
import collections
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
"""
#Setting Matplotlib Graph Visual Prameters
plt.rcParams["axes.labelsize"] = 16.
plt.rcParams["xtick.labelsize"] = 14.
plt.rcParams["ytick.labelsize"] = 14.
plt.rcParams["legend.fontsize"] = 12.
plt.rcParams["figure.figsize"] = [15., 6.]


#Loading Dataset
battles=pd.read_csv("database/battles.csv")
character_deaths=pd.read_csv("database/character-deaths.csv")
character_predictions=pd.read_csv("database/character-predictions.csv")

#Pre-processing Data

#Adding Attributes for Analysis
battles.loc[:, "defender_count"] = (4 - battles[["defender_1", "defender_2", "defender_3", "defender_4"]].isnull().sum(axis = 1))   #Number of major houses on defending side
battles.loc[:, "attacker_count"] = (4 - battles[["attacker_1", "attacker_2", "attacker_3", "attacker_4"]].isnull().sum(axis = 1))   #Number of major houses on attacking side
battles.loc[:, "att_comm_count"] = [len(x) if type(x) == list else np.nan for x in battles.attacker_commander.str.split(",")]       #Number of commanders on attacking side
character_predictions.loc[:, "no_of_books"] = character_predictions[[x for x in character_predictions.columns if x.startswith("book")]].sum(axis = 1)   #Number of books a character appeared in

#Filling up blank datacells
battles.iat[37,13]='win'
battles.loc[battles["name"]=="Siege of Winterfell","battle_type"]='siege'
battles.loc[battles["name"]=="Siege of Winterfell","major_death"]=0
battles.loc[battles["name"]=="Siege of Winterfell","major_capture"]=0

battles['attacker_king'] = battles['attacker_king'].fillna('Without a king')   
battles['defender_king'] = battles['defender_king'].fillna('Without a king')
  
battles['defender_1'] = battles['defender_1'].fillna('common people')

battles['attacker_commander'] = battles['attacker_commander'].fillna('Without a commander')   
battles['defender_commander'] = battles['defender_commander'].fillna('Without a commander')
  
battles['location'] = battles['location'].fillna('Dont know where')

#Doing some formatting
character_deaths.head()
character_deaths.shape[0]
#character_deaths.info()
character_deaths.shape[0] - character_deaths['Death Chapter'].dropna().shape[0]
#character_deaths.describe()
#character_deaths.columns
character_deaths = character_deaths.rename(columns=lambda x: x.replace(' ', '_').lower())
character_deaths.columns

#Adding Features-Compute Battle Size for statistics
#Only if we have to full the size of battle
#attacker_size_mid = battles['attacker_size'].median()  
#defender_size_mid = battles['defender_size'].median()
for i in range(1,len(battles)):
    if  np.isnan(battles.iloc[i,17]) and np.isnan(battles.iloc[i,18]):
        continue
    elif np.isnan(battles.iloc[i,17]):
        battles.iat[i,17] = battles.iat[i,18]
    elif np.isnan(battles.iloc[i,18]):
        battles.iat[i,18] = battles.iat[i,17]
        
battles['battle_size'] = battles['attacker_size'] + battles['defender_size']
