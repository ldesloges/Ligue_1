import pandas as pd
import numpy as np
from scipy.stats import poisson



data = pd.read_csv('../data/L1_24_25.csv')

data = data[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
data = data.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})

Home_goals_mean=data['HomeGoals'].mean()
Away_goals_mean=data['AwayGoals'].mean()

teams=data['HomeTeam'].unique()

Capacity={}


for team in teams:
    home_matches = data[data['HomeTeam'] == team]
    away_matches = data[data['AwayTeam'] == team]

    Home_goals_mean_team=home_matches['HomeGoals'].mean()
    Away_goals_mean_team=away_matches['AwayGoals'].mean()
    Home_taken_mean_team=home_matches['AwayGoals'].mean()
    Away_taken_mean_team=away_matches['HomeGoals'].mean()
    
    Home_goals_capacity=Home_goals_mean_team/Home_goals_mean
    Away_goals_capacity=Away_goals_mean_team/Away_goals_mean
    Home_taken_capacity=Home_taken_mean_team/Away_goals_mean
    Away_taken_capacity=Away_taken_mean_team/Home_goals_mean

    Capacity[team] = {
    'Home_goals_capacity': Home_goals_capacity,
    'Away_goals_capacity': Away_goals_capacity,           
    'Home_taken_capacity': Home_taken_capacity,         
    'Away_taken_capacity': Away_taken_capacity,             
    }

classement = {
    'Equipe': teams,
    'Pts': 0,           
    'Joués': 0,         
    'G': 0,             
    'N': 0,              
    'P': 0,             
    'BP': 0,           
    'BC': 0,            
    'Diff': 0           
}
classement = pd.DataFrame(classement).set_index('Equipe')



def simuler_match_poisson(equipe_dom, equipe_ext):
    
   lambda_dom=Home_goals_mean*Capacity[equipe_dom]['Home_goals_capacity']*Capacity[equipe_ext]['Away_taken_capacity']
   lambda_ext=Away_goals_mean*Capacity[equipe_ext]['Away_goals_capacity']*Capacity[equipe_dom]['Home_taken_capacity']

   buts_dom = poisson.rvs(lambda_dom, size=1)[0]
   buts_ext = poisson.rvs(lambda_ext, size=1)[0]

   return buts_dom,buts_ext


def mettre_a_jour_classement(classement, equipe_dom, equipe_ext):


    buts_dom=simuler_match_poisson(equipe_dom,equipe_ext)[0]
    buts_ext=simuler_match_poisson(equipe_dom,equipe_ext)[1]
    
    
    classement.loc[equipe_dom, 'Joués'] += 1
    classement.loc[equipe_ext, 'Joués'] += 1
    
 
    classement.loc[equipe_dom, 'BP'] += buts_dom
    classement.loc[equipe_dom, 'BC'] += buts_ext
    
   
    classement.loc[equipe_ext, 'BP'] += buts_ext
    classement.loc[equipe_ext, 'BC'] += buts_dom
    
    
    if buts_dom > buts_ext:
    
        classement.loc[equipe_dom, 'Pts'] += 3
        classement.loc[equipe_dom, 'G'] += 1
        classement.loc[equipe_ext, 'P'] += 1
        
    elif buts_dom < buts_ext:
        
        classement.loc[equipe_ext, 'Pts'] += 3
        classement.loc[equipe_ext, 'G'] += 1
        classement.loc[equipe_dom, 'P'] += 1
        
    else: 
        
        classement.loc[equipe_dom, 'Pts'] += 1
        classement.loc[equipe_ext, 'Pts'] += 1
        classement.loc[equipe_dom, 'N'] += 1
        classement.loc[equipe_ext, 'N'] += 1
        
    classement['Diff'] = classement['BP'] - classement['BC']
    
    return classement 


def afficher_classement_final(classement):
    
    
    criteres_de_tri = ['Pts', 'Diff', 'BP']
    
    ordre_decroissant = [False, False, False]
    
    classement_final = classement.sort_values(
        by=criteres_de_tri,
        ascending=ordre_decroissant
    )
    
    return classement_final


def simuler_saison_25_26():
    for teamA in teams:
        for teamB in teams:
            mettre_a_jour_classement(classement,teamA,teamB)
    print(afficher_classement_final(classement))


print(simuler_match_poisson('Paris SG' , 'Nantes'))