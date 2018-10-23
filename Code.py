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

#Battle Analysis

#1. Battle Type
plt.figure(figsize=[10,6])
battles.battle_type.value_counts().plot(kind='bar')
plt.title("Battle Type") 
plt.ylabel("count") 
plt.show()

#2. Win/Defeat in Battle
kingd = battles['defender_king']
kinga = battles['attacker_king']
king = kinga.append(kingd)
king = king.drop_duplicates()

l = [0]
l = l * len(king)
w = dict(zip(king,l))
d = dict(zip(king,l))

for i in range(0,len(battles)):
    if  battles.iloc[i,13]=='win':
        w[battles.iloc[i,3]]+=1
        d[battles.iloc[i,4]]+=1
    else:
        w[battles.iloc[i,4]]+=1
        d[battles.iloc[i,3]]+=1
        
plt.figure(figsize=[15,6])
plt.bar(range(len(w)), list(w.values()), align='center')
plt.xticks(range(len(w)), list(w.keys()), rotation=90)
plt.title("Number of Battles Win")
plt.show()
plt.figure(figsize=[15,6])
plt.bar(range(len(d)), list(d.values()), align='center')
plt.xticks(range(len(d)), list(d.keys()), rotation=90)
plt.title("Number of Battles Lost")
plt.show()

#3. Army Size of House
house = dict(zip(king,l))

for i in range(0,len(battles)):
    if battles.iloc[i,17] / 1 == battles.iloc[i,17]:
        house[battles.iloc[i,3]] += battles.iloc[i,17]
    if battles.iloc[i,18] / 1 == battles.iloc[i,18]:
        house[battles.iloc[i,4]] += battles.iloc[i,18]
    print(i,house)

plt.figure(figsize=[15,6])
plt.bar(range(len(house)), list(house.values()), align='center')
plt.xticks(range(len(house)), list(house.keys()), rotation=90)
plt.title("Army Size of Each House")
plt.show()

#11. Death By Relation
data = character_predictions.groupby(["boolDeadRelations", "isAlive"]).count()["S.No"].unstack().copy(deep = True)
p = data.div(data.sum(axis = 1), axis = 0).plot.barh(stacked = True, rot = 0, width = .5)
_ = p.set_xlim([0, 1]), p.set(yticklabels = ["No", "Yes"], xticklabels = "", xlabel = "Proportion of Dead vs. Alive", ylabel = "Has Dead Relations"), p.legend(["Dead", "Alive"])

#7. Major Death/Capture Events By Region
data = battles.groupby("region").sum()[["major_death", "major_capture"]]
p = pd.concat([data, battles.region.value_counts().to_frame()], axis = 1).sort_values("region", ascending = False).copy(deep = True).plot.bar(color = [sns.color_palette()[1], "grey", "darkblue"], rot = 0)
_ = p.set(xlabel = "Region", ylabel = "No. of Events"), p.legend(["Major Deaths", "Major Captures", "No. of Battles"], fontsize = 12.)

#8. Major Death/Capture By Year
p = battles.groupby('year').sum()[["major_death", "major_capture"]].plot.bar(rot = 0)
_ = p.set(xlabel = "Year", ylabel = "No. of Death/Capture Events", ylim = (0, 9)), p.legend(["Major Deaths", "Major Captures"])

#12. Appearance in No. of Books before Death
data = character_predictions.groupby(["no_of_books", "isAlive"]).count()["S.No"].unstack().copy(deep = True)
p = data.div(data.sum(axis = 1), axis = 0).plot.barh(stacked = True, rot = 0, figsize = (15, 8), width = .5)
_ = p.set(xticklabels = "", xlim = [0, 1], ylabel = "No. of Books", xlabel = "Proportion of Dead vs. Alive"), p.legend(["Dead", "Alive"], loc = "upper right", ncol = 2, borderpad = -.15)


#4. Commanders in each army
p = sns.boxplot("att_comm_count", "attacker_king", data = battles, saturation = .6, fliersize = 10., palette = ["lightgray", sns.color_palette()[1], "grey", "darkblue"])
_ = p.set(xlabel = "No. of Attacker Commanders", ylabel = "Attacker King", xticks = range(8))
