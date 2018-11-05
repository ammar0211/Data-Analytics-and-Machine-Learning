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


#9. print("How does culture relate to survival?")
cult = {
    'Summer Islands': ['summer islands', 'summer islander', 'summer isles'],
    'Ghiscari': ['ghiscari', 'ghiscaricari',  'ghis'],
    'Asshai': ["asshai'i", 'asshai'],
    'Lysene': ['lysene', 'lyseni'],
    'Andal': ['andal', 'andals'],
    'Braavosi': ['braavosi', 'braavos'],
    'Dornish': ['dornishmen', 'dorne', 'dornish'],
    'Myrish': ['myr', 'myrish', 'myrmen'],
    'Westermen': ['westermen', 'westerman', 'westerlands'],
    'Westerosi': ['westeros', 'westerosi'],
    'Stormlander': ['stormlands', 'stormlander'],
    'Norvoshi': ['norvos', 'norvoshi'],
    'Northmen': ['the north', 'northmen'],
    'Free Folk': ['wildling', 'first men', 'free folk'],
    'Qartheen': ['qartheen', 'qarth'],
    'Reach': ['the reach', 'reach', 'reachmen'],
}

def get_cult(value):
    value = value.lower()
    v = [k for (k, v) in cult.items() if value in v]
    return v[0] if len(v) > 0 else value.title()
character_predictions.loc[:, "culture"] = [get_cult(x) for x in character_predictions.culture.fillna("")]
data1 = character_predictions.groupby(["culture", "isAlive"]).count()["S.No"].unstack().copy(deep = True)
data1.loc[:, "total"]= data1.sum(axis = 1)
p = data1[data1.index != ""].sort_values("total")[[0, 1]].plot.barh(stacked = True, rot = 0, figsize = (14, 12),)
_ = p.set(xlabel = "No. of Characters", ylabel = "Culture"), p.legend(["Dead", "Alive"], loc = "lower right")


#5. Army Size Differnece for wning side in Battles
data = battles.dropna(subset = ["attacker_size", "defender_size"]).copy(deep = True)
data = pd.concat([(data.attacker_size - data.defender_size).to_frame(), battles[["attacker_outcome"]]], axis = 1, join = "inner")
data = data[data[0] != 0]
p = data[0].plot.barh(figsize = (12, 8), width = .8, color = [sns.color_palette()[0] if x == "win" else sns.color_palette()[2] if x == "loss" else "white" for x in data.attacker_outcome.values])
_ = p.legend(handles = [mpatches.Patch(color = sns.color_palette()[0], label = "Victory", aa = True), mpatches.Patch(color = sns.color_palette()[2], label = "Loss", aa = True)])
_ = p.axvline(0, color = 'k'), p.set(yticklabels = battles.name.iloc[data.index].values, xlabel = "Difference in Army Size (attacker_size - defender_size)", ylabel = "Battle Name")


#6. Region of Battle
plt.figure(figsize=[10,4])
battles.region.value_counts().plot(kind='bar')
plt.title("Region Distribution") 
plt.ylabel("count")
plt.title("Regions involved in Battles") 
plt.show()


#9. Proportion of Death by Allegiance
character_deaths_by_chapter = character_deaths.groupby('death_chapter' ).count()
def plot_it(x=[], y=[], kind="plot", title="Your chart", xlabel="x-axis", ylabel="y-axis"):
    """ This function plot different type of charts depending on args value. """
    if kind == "plot":
        plt.plot(x, y)
    elif kind == "scatter":
        plt.scatter(x, y)
    elif kind == "bar":
        plt.bar(x, y, color=np.random.rand(256,3))
    else:
        raise ValueError(kind + ' is not a supported type of chart.')
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 26
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
character_deaths['allegiances'].unique()
character_deaths['allegiances'] = character_deaths['allegiances'].apply(lambda x: x.replace('House ', '').lower())
df_characters_by_allegiances = character_deaths.groupby('allegiances').count()
df_characters_by_allegiances[['name']]
character_deaths['death_chapter'] = character_deaths['death_chapter'].fillna('none')
df_dead_characters = character_deaths[character_deaths['death_chapter'] != 'none'].copy()
character_deaths_by_allegiances = df_dead_characters.groupby('allegiances').count()
character_deaths_by_allegiances[['name']]
character_deaths_by_allegiances['death_proportion'] = character_deaths_by_allegiances['name'] / df_characters_by_allegiances['name']
plot_it(character_deaths_by_allegiances.index, character_deaths_by_allegiances['death_proportion'], 'bar', 'Number of deaths by allegiances', 'Allegiances', 'Deaths')

#10. Death By Number of Chapters
plot_it(character_deaths_by_chapter.index, character_deaths_by_chapter['name'], 'plot', 'Evolution of the number of deaths through chapters', 'Chapter Number', 'Deaths')

#15. print("How many chapters a character takes to die ?")
df_dead_characters['alive_chapters'] = df_dead_characters['death_chapter'] - df_dead_characters['book_intro_chapter']
df_dead_characters = df_dead_characters[df_dead_characters['alive_chapters'] >= 0]
df_dead_characters[['alive_chapters']].head()
print("Median of Chapter Character was alive: ",df_dead_characters['alive_chapters'].median())
print("Character with most chapter before death: ",df_dead_characters.sort_values('alive_chapters', ascending=False).iloc[0])


#--------------------- character_predictions Cleansing -----------------------------------------

#-------- culture induction -------
cult = {
    'Summer Islands': ['summer islands', 'summer islander', 'summer isles'],
    'Ghiscari': ['ghiscari', 'ghiscaricari',  'ghis'],
    'Asshai': ["asshai'i", 'asshai'],
    'Lysene': ['lysene', 'lyseni'],
    'Andal': ['andal', 'andals'],
    'Braavosi': ['braavosi', 'braavos'],
    'Dornish': ['dornishmen', 'dorne', 'dornish'],
    'Myrish': ['myr', 'myrish', 'myrmen'],
    'Westermen': ['westermen', 'westerman', 'westerlands'],
    'Westerosi': ['westeros', 'westerosi'],
    'Stormlander': ['stormlands', 'stormlander'],
    'Norvoshi': ['norvos', 'norvoshi'],
    'Northmen': ['the north', 'northmen'],
    'Free Folk': ['wildling', 'first men', 'free folk'],
    'Qartheen': ['qartheen', 'qarth'],
    'Reach': ['the reach', 'reach', 'reachmen'],
}

def get_cult(value):
    value = value.lower()
    v = [k for (k, v) in cult.items() if value in v]
    return v[0] if len(v) > 0 else value.title()

character_predictions.loc[:, "culture"] = [get_cult(x) for x in character_predictions.culture.fillna("")]


#-------- culture induction -------

character_predictions.drop(["S.No","name", "alive", "pred", "plod", "isAlive", "dateOfBirth", "DateoFdeath"], 1, inplace = True)

character_predictions.loc[:, "title"] = pd.factorize(character_predictions.title)[0]
character_predictions.loc[:, "culture"] = pd.factorize(character_predictions.culture)[0]
character_predictions.loc[:, "mother"] = pd.factorize(character_predictions.mother)[0]
character_predictions.loc[:, "father"] = pd.factorize(character_predictions.father)[0]
character_predictions.loc[:, "heir"] = pd.factorize(character_predictions.heir)[0]
character_predictions.loc[:, "house"] = pd.factorize(character_predictions.house)[0]
character_predictions.loc[:, "spouse"] = pd.factorize(character_predictions.spouse)[0]

character_predictions.fillna(value = -1, inplace = True)
''' $$ The code below usually works as a sample equilibrium. However in this case,
 this equilibirium actually decrease our accuracy, all because the original 
prediction character_predictions was released without any sample balancing. $$

character_predictions = character_predictions[character_predictions.actual == 0].sample(350, random_state = 62).append(character_predictions[character_predictions.actual == 1].sample(350, random_state = 62)).copy(deep = True).astype(np.float64)

'''
Y = character_predictions.actual.values

Ocharacter_predictions = character_predictions.copy(deep=True)

character_predictions.drop(["actual"], 1, inplace = True)

#------------------ Feature Correlation ---------------------------------------

sns.heatmap(character_predictions.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #character_predictions.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(30,20)
plt.show()

#------------------ Predicting ------------------------------------------------

#---------- 1 RandomForest -----------------
''' ATTENTION: This rf algorithm achieves 99%+ accuracy, this is because the \
    original predictor-- the document releaser use exactly the same algorithm to predict!
'''
from sklearn.ensemble import RandomForestClassifier


random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(character_predictions, Y)

print('RandomForest Accuracy(original): ',random_forest.score(character_predictions, Y))

#---------- 2 DecisionTree -----------------

from sklearn.tree import DecisionTreeClassifier

DT=DecisionTreeClassifier()

DT.fit(character_predictions,Y)

print('DecisionTree Accuracy(original)： ',DT.score(character_predictions, Y))

