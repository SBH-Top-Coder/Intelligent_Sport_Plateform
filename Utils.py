import numpy as np
from tqdm import tqdm
import json
import pandas as pd
import random
Big_Teams = ['Borussia Dortmund',
                 'Juventus', 'Zenit', 'Porto', 'Spartak Moscow',
                 'Ajax',
                 'Atletico Madrid', 'Fenerbahçe',
                 'Chelsea', 'Benfica',
                 'Athletic Club', 'Genk',
                 'Real Madrid',
                 'Atalanta', 'Anderlecht',
                 'Barcelona', 'Beşiktaş',
                 'PSG',
                 'Liverpool',
                 'Bayern Munich',
                 'Manchester City', 'Internazionale']
medium_Teams = ['Leicester City',
                    'Arsenal',
                    'Valencia',
                    'Everton',
                    'Real Sociedad',
                    'West Ham United',
                    'Villarreal',
                    'Bayer Leverkusen',
                    'Napoli',
                    'Lazio']
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

def load (path):
  df = pd.read_csv(path)
  return df

def prepare_data (matchesDF1,matchesDF2,matchesDF3) :
  allMatchesDF = pd.concat([matchesDF1,matchesDF2,matchesDF3])
  allMatchesDF = allMatchesDF.drop_duplicates(["homeTeam","awayTeam","matchDate"])
  return allMatchesDF

def convert_label (x) :
  if int(x.split("-")[0]) > int(x.split("-")[1])  :
    return 2
  elif int(x.split("-")[0]) < int(x.split("-")[1]) :
    return 0
  else :
    return 1

def convert_home_Last(x):
    if x == "L":
        return 0
    elif x == "D":
        return 1
    elif x == "W":
        return 2

def convert_away_Last(x):
  if x == "L": return 2
  elif x== "D" : return 1
  elif x == "W": return 0


# def data_augmentation(allMatchesDF):
#     allMatchesDF_augmented = pd.DataFrame([], columns=allMatchesDF.columns)
#     for idx, row in tqdm(allMatchesDF.iterrows()):
#         for i in range(row['homeWins']):
#             row[3] = 3
#             df_tmp = pd.DataFrame([tuple(row.values)],
#                                   columns=allMatchesDF.columns)
#             allMatchesDF_augmented = allMatchesDF_augmented.append(df_tmp)
#
#         for i in range(row['homeDraws']):
#             row[3] = 1
#             df_tmp = pd.DataFrame([tuple(row.values)],
#                                   columns=allMatchesDF.columns)
#             allMatchesDF_augmented = allMatchesDF_augmented.append(df_tmp)
#
#         for i in range(row['homeLosses']):
#             row[3] = 0
#             df_tmp = pd.DataFrame([tuple(row.values)],
#                                   columns=allMatchesDF.columns)
#             allMatchesDF_augmented = allMatchesDF_augmented.append(df_tmp)
#     return allMatchesDF_augmented

def patch_extractor (path):
  f = open (path, "r")
  teamPatches = json.loads(f.read())
  f.close()
  return teamPatches

def prepare_column_names (allMatchesDF):
  columns = list(allMatchesDF.columns)
  full_columns =  columns + ['overallRating_home','attackRating_home','midRating_home','defenceRating_home','internationlPrestige_home','domesticPrestige_home','transferBudget_home','averageStartingAge_home','averageAllAge_home']
  return full_columns

# équipét dawri rousi mafamch fl json ... Tsaraf fihom nchalah
def extend_data (full_columns) :
  allMatchesDF_extented = pd.DataFrame([], columns=full_columns)
  not_found = []
  for idx,row in tqdm(allMatchesDF.iterrows()) :
    try :
      attributes = teamPatches[row['homeTeam']][row['season']][0]
    except :
      not_found.append(row['homeTeam'])
    row['overallRating_home'],row['attackRating_home'],row['midRating_home'],row['defenceRating_home'],row['internationlPrestige_home'],row['domesticPrestige_home'],row['transferBudget'],row['averageStartingAge_home'],row['averageAllAge_home'] = attributes[0],attributes[1],attributes[2],attributes[3],attributes[4],attributes[5],float(attributes[6][1:][:-1]),attributes[7],attributes[8]
    df_tmp =  pd.DataFrame([tuple(row.values)],
                                    columns=full_columns)
    allMatchesDF_extented = allMatchesDF_extented.append(df_tmp)
    return allMatchesDF_extented,not_found

# Tjm tzid tba3wél fiha kima thbb ...........
def Home_Form (df):
  HF = []
  for idx,row in df.iterrows() :
    form = row.values
    W = 0
    for x in form  :
      if x == 3 :
        W = W + 1
    if (W >= 4 ) :
      HF.append(30)
    # elif ((W == 3) or (W==2 ) ):
    #   HF.append(10)
    else :
      HF.append(0)
  return HF

# def  Away_Form (df):
#   AF = []
#   for idx,row in df.iterrows() :
#     L = row .values
#     W = 0
#     for x in L :
#       if x == 0 :
#         W = W + 1
#     if (W >= 4 ) :
#       AF.append(0)
#     elif ((W == 3) or (W==2 ) ):
#       AF.append(10)
#     else :
#       AF.append(30)
#   return AF

def victory_pourcentage (df):
  pourcentage = [ row ["homeWins"]/row['matchesPlayed'] for idx,row in df.iterrows()]
  return pourcentage


def big_home_team(df):
    big = []
    for idx, row in df.iterrows():
        if (row['homeTeam'] in Big_Teams):
            big.append(12)
        elif (row['homeTeam'] in medium_Teams):
            big.append(4)
        else:
            big.append(0)
    return big

# def big_away_team (df):
#   big = []
#   for idx,row in df.iterrows() :
#     if (row['awayTeam'] in Big_Teams ) :
#       big.append(0)
#     else :
#       big.append(10)
#   return big

def force_attaque_home (allMatchesDF):
  Attaque_home = []
  sum = 0
  for (idx,df ) in allMatchesDF.groupby(['homeTeam','season']) :
    for x in df['res'] :
      sum = sum + int(x.split('-')[0])
    Attaque_home.append((idx , sum/len(df)))
    sum  = 0
  return Attaque_home

def force_defence_home (allMatchesDF):
  Defense_Home= []
  sum = 0
  for (idx,df ) in allMatchesDF.groupby(['homeTeam','season']) :
    for x in df['res'].values :
      sum = sum + int(x.split('-')[1])
    Defense_Home.append((idx , sum/len(df)))
    sum  = 0
  return Defense_Home

def force_Defense_away (allMatchesDF):
  Defense_away= []
  sum = 0
  for (idx,df ) in allMatchesDF.groupby(['awayTeam','season']) :
    for x in df['res'].values :
      sum = sum + int(x.split('-')[0])
    Defense_away.append((idx , sum/len(df)))
    sum  = 0
  return Defense_away

def force_attaque_away (allMatchesDF):
  Attaque_away= []
  sum = 0
  for (idx,df ) in allMatchesDF.groupby(['awayTeam','season']) :
    for x in df['res'].values :
      sum = sum + int(x.split('-')[1])
    Attaque_away.append((idx , sum/len(df)))
    sum  = 0
  return Attaque_away

def force_home (allMatchesDF):
  Pts_home = []
  sum = 0
  for (idx,df ) in allMatchesDF.groupby(['homeTeam','season']) :
    for x in df['res'].values :
      if ( int(x.split('-')[0]) >  int(x.split('-')[1])) :
          sum = sum + 3
      elif ( int(x.split('-')[0]) ==  int(x.split('-')[1])) :
          sum = sum + 1
      else :
          sum = sum + 0
    Pts_home.append((idx , sum/len(df)))
    sum  = 0
  return Pts_home

def force_away (allMatchesDF):
  Pts_away = []
  sum = 0
  for (idx,df ) in allMatchesDF.groupby(['awayTeam','season']) :
    for x in df['res'].values :
      if ( int(x.split('-')[1]) >  int(x.split('-')[0])) :
          sum = sum + 3
      elif ( int(x.split('-')[0]) ==  int(x.split('-')[1])) :
          sum = sum + 1
      else :
          sum = sum + 0
    Pts_away.append((idx , sum/len(df)))
    sum  = 0
  return Pts_away

def splitting_data (allMatchesDF):
  allMatchesDF = allMatchesDF.sample(frac=1).reset_index(drop=True)
  train,test = allMatchesDF[:5400] , allMatchesDF[5400:]
  return train,test
