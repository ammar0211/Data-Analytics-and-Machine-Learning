# Content

Of course, it goes without saying that this dataset contains spoilers ;)

This dataset combines three sources of data, all of which are based on information from the book series.

* Firstly, there is battles.csv which contains Chris Albon's "The War of the Five Kings" Dataset. Its a great collection of all of the battles in the series.

* Secondly we have character-deaths.csv from Erin Pierce and Ben Kahle. This dataset was created as a part of their Bayesian Survival Analysis.

* Finally we have a more comprehensive character dataset with character-predictions.csv. It includes their predictions on which character will die.

# Acknowledgements</b>

* Firstly, there is battles.csv which contains Chris Albon's "The War of the Five Kings" Dataset, which can be found here: https://github.com/chrisalbon/war_of_the_five_kings_dataset . Its a great collection of all of the battles in the series.

* Secondly we have character-deaths.csv from Erin Pierce and Ben Kahle. This dataset was created as a part of their Bayesian Survival Analysis which can be found here: http://allendowney.blogspot.com/2015/03/bayesian-survival-analysis-for-game-of.html

* Finally we have a more comprehensive character dataset with character-predictions.csv. This comes from the team at A Song of Ice and Data who scraped it from http://awoiaf.westeros.org/ . It also includes their predictions on which character will die, the methodology of which can be found here: https://got.show/machine-learning-algorithm-predicts-death-game-of-thrones

# Database Columns

## battles.csv

This csv contains information about all of the battles in game of thrones. It is 38x25.

```
* name: The name of the battle.
* year: The year of the battle.
* battle_number: A unique ID number for the battle.
* attacker_king: The attacker's king. A slash indicators that the king charges over the course of the war. For example, "Joffrey/Tommen Baratheon" is coded as such because one king follows the other in the Iron Throne.
* defender_king: The defender's king.
* attacker_1: Major house attacking.
* attacker_2: Major house attacking.
* attacker_3: Major house attacking.
* attacker_4: Major house attacking.
* defender_1: Major house defending.
* defender_2: Major house defending.
* defender_3: Major house defending.
* defender_4: Major house defending.
* attacker_outcome: The outcome from the perspective of the attacker. Categories: win, loss, draw.
* battle_type: A classification of the battle's primary type. Categories: pitched_battle: Armies meet in a location and fight. This is also the baseline category. ambush: A battle where stealth or subterfuge was the primary means of attack. siege: A prolonged of a fortied position. razing: An attack against an undefended position
* major_death: If there was a death of a major figure during the battle.
* major_capture: If there was the capture of the major figure during the battle.
* attacker_size: The size of the attacker's force. No distinction is made between the types of soldiers such as cavalry and footmen.
* defender_size: The size of the defenders's force. No distinction is made between the types of soldiers such as cavalry and footmen.
* attacker_commander: Major commanders of the attackers. Commander's names are included without honorific titles and commanders are separated by commas.
* defender_commander: Major commanders of the defender. Commander's names are included without honorific titles and commanders are separated by commas.
* summer: Was it summer?
* location: The location of the battle.
* region: The region where the battle takes place. Categories: Beyond the Wall, The North, The Iron Islands, The Riverlands, The Vale of Arryn, The Westerlands, The Crownlands, The Reach, The Stormlands, Dorne
noteCoding notes regarding individual observations.

```


## character-deaths.csv

This csv contains information about characters and when they died. It is 917x13.

```
* Name: character name
* Allegiances: character house
* Death Year: year character died
* Book of Death: book character died
* Death Chapter: book character died in
* Book Intro Chapter: chapter character was introduced in
* Gender: 1 is male, 0 is female
* Nobility: 1 is nobel, 0 is a commoner
* GoT: Appeared in first book
* CoK: Appeared in second book
* SoS: Appeared in third book
* FfC: Appeared in fourth book
* DwD: Appeared in fifth book
```

## character-predictions.csv

This csv contains information about all the characters' prediction of death in game of thrones. It is 1946x33.

```

* S.No
* actual
* pred
* alive
* plod
* name
* title
* male
* culture
* dateOfBirth
* DateoFdeath
* mother
* father
* heir
* house
* spouse
* book1
* book2
* book3
* book4
* book5
* isAliveMother
* isAliveFather
* isAliveHeir
* isAliveSpouse
* isMarried
* isNoble
* age
* numDeadRelations
* boolDeadRelations
* isPopular
* popularity
* isAlive
```
