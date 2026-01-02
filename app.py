import pandas as pd
import numpy as np
from scipy.stats import poisson
from matplotlib import pyplot as plt
import streamlit as st

#Calendrier
calendrier_25_26=pd.read_csv(f'data/calendrier_25_26.csv')
calendrier_25_26=calendrier_25_26[['wk','HomeTeam','AwayTeam']]


#Création de la DataFrame
tous_les_matchs = []
for i in range(9):
    df = pd.read_csv(f'data/L1_{16+i}_{17+i}.csv')
    df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
    tous_les_matchs.append(df)

#Concaténation des datas
data_historique = pd.concat(tous_les_matchs, ignore_index=True)

HOME_GOALS_MEAN_GLOBAL = data_historique['HomeGoals'].mean()
AWAY_GOALS_MEAN_GLOBAL = data_historique['AwayGoals'].mean()

Ligue1_25_26=pd.read_csv(f'data/L1_25_26.csv')
toutes_equipes = pd.concat([Ligue1_25_26['HomeTeam'], Ligue1_25_26['AwayTeam']]).unique()
TEAMS = list(toutes_equipes)

#Attribution de apacité a marquer et a ecaisser a domicile et a l'exterieur
Capacity={}

def promotion_L1(team_promu):
    Capacity[team_promu] = {
    'Home_goals_capacity': 1,
    'Away_goals_capacity': 1,           
    'Home_taken_capacity': 1,         
    'Away_taken_capacity': 1,             
    }

for team in TEAMS:
    if not team in set(data_historique['HomeTeam'].unique()):
        promotion_L1(team)
    else:
        home_matches = data_historique[data_historique['HomeTeam'] == team]
        away_matches = data_historique[data_historique['AwayTeam'] == team]

        Home_goals_mean_team=home_matches['HomeGoals'].mean()
        Away_goals_mean_team=away_matches['AwayGoals'].mean()
        Home_taken_mean_team=home_matches['AwayGoals'].mean()
        Away_taken_mean_team=away_matches['HomeGoals'].mean()
        
        Home_goals_capacity=Home_goals_mean_team/HOME_GOALS_MEAN_GLOBAL
        Away_goals_capacity=Away_goals_mean_team/AWAY_GOALS_MEAN_GLOBAL
        Home_taken_capacity=Home_taken_mean_team/AWAY_GOALS_MEAN_GLOBAL
        Away_taken_capacity=Away_taken_mean_team/HOME_GOALS_MEAN_GLOBAL

        Capacity[team] = {
        'Home_goals_capacity': Home_goals_capacity,
        'Away_goals_capacity': Away_goals_capacity,           
        'Home_taken_capacity': Home_taken_capacity,         
        'Away_taken_capacity': Away_taken_capacity,             
        }

#Création du classement initial
classement = {
    'Equipe': TEAMS,
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


#Simulation d'un match
def simuler_match_poisson(equipe_dom, equipe_ext):
    
   lambda_dom=HOME_GOALS_MEAN_GLOBAL*Capacity[equipe_dom]['Home_goals_capacity']*Capacity[equipe_ext]['Away_taken_capacity']
   lambda_ext=AWAY_GOALS_MEAN_GLOBAL*Capacity[equipe_ext]['Away_goals_capacity']*Capacity[equipe_dom]['Home_taken_capacity']

   buts_dom = poisson.rvs(lambda_dom, size=1)[0]
   buts_ext = poisson.rvs(lambda_ext, size=1)[0]

   return buts_dom,buts_ext

#mise a jour du classemement apres un match
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

#Mettre le classement dans le bonne ordre
def afficher_classement_final(classement):
    
    
    criteres_de_tri = ['Pts', 'Diff', 'BP']
    
    ordre_decroissant = [False, False, False]
    
    classement_final = classement.sort_values(
        by=criteres_de_tri,
        ascending=ordre_decroissant
    )
    
    return classement_final


#Simulation d'une journée
def simuler_wk(j, classement, calendrier):

    matchs_journee = calendrier[calendrier['wk'] == j]

    for index, match in matchs_journee.iterrows():
        equipe_dom = match['HomeTeam']
        equipe_ext = match['AwayTeam']
        
        classement = mettre_a_jour_classement(classement, equipe_dom, equipe_ext) 
    
    return afficher_classement_final(classement)

#Simulation d'une saison
def simuler_saison_25_26():
    for teamA in TEAMS:
        for teamB in TEAMS:
            mettre_a_jour_classement(classement,teamA,teamB)
    print(afficher_classement_final(classement))



#Obtention des rangs par equipes aprés chaque journée
def simuler_saison_et_tracker_rangs(calendrier_df):
    print("Simulation de la saison en cours ...")

    classement = {
        'Equipe': TEAMS,
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
    
    liste_classement= []

    for j in range(1, 35):
        classement_mis_a_jour = simuler_wk(j, classement, calendrier_df)
        
        liste_classement.append(classement_mis_a_jour.copy())
        
        classement = classement_mis_a_jour.copy() 
        
    return liste_classement



#Création liste des rang par equipe
def creer_historique_par_club(liste_classement, TEAMS):

    historique_par_club = {}
    
    for team in TEAMS:
        historique_par_club[team] = []
        
    for classement in liste_classement:

        for team in TEAMS:
            position_index_zero = classement.index.get_loc(team)
                
            rang = position_index_zero + 1
            historique_par_club[team].append(rang)

    return historique_par_club


journees=[i for i in range(1,35)]
def tracer_evolution_classement(TEAMS):
    

    historique_par_club=creer_historique_par_club(simuler_saison_et_tracker_rangs(calendrier_25_26),TEAMS)
    plt.figure(figsize=(15, 8))
    
    for club, rangs in historique_par_club.items():
        plt.plot(journees, rangs, label=club, linewidth=2)
    
    
    plt.gca().invert_yaxis()
    
    plt.axhline(1, color='gold', linestyle='--', alpha=0.6, label='Champion')
    plt.axhline(3, color='blue', linestyle='--', alpha=0.4, label='Ligue des Champions')
    plt.axhline(17, color='red', linestyle='--', alpha=0.4, label='Barrages/Relégation')
    
    plt.title('Évolution du Rang de Classement (Simulation Séquentielle L1)', fontsize=16)
    plt.xlabel('Journée', fontsize=12)
    plt.ylabel('Rang de Classement', fontsize=12)
    
    plt.yticks(np.arange(1, len(TEAMS) + 1, 1)) 
    plt.xticks(np.arange(1, len(journees) + 1, 2)) 
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.tight_layout()
    plt.show()

#tracer_evolution_classement(creer_historique_par_club(simuler_saison_et_tracker_rangs(calendrier_25_26),TEAMS))

import plotly.express as px

import base64

def encoder_svg_local(chemin_fichier):
    with open(chemin_fichier, "rb") as f:
        image_data = f.read()
        base64_string = base64.b64encode(image_data).decode("utf-8")
        return f"data:image/svg+xml;base64,{base64_string}"



LOGOS = {
    "Paris SG": encoder_svg_local("include/PSG.svg"),
    "Rennes": encoder_svg_local("include/Rennes.svg"),
    "Lens": encoder_svg_local("include/Lens.svg"),
    "Monaco": encoder_svg_local("include/Monaco.svg"),
    "Nice": encoder_svg_local("include/Nice.svg"),
    "Brest": encoder_svg_local("include/Brest.svg"),
    "Angers": encoder_svg_local("include/Angers.svg"),
    "Auxerre": encoder_svg_local("include/Auxerre.svg"),
    "Metz": encoder_svg_local("include/Metz.svg"),
    "Nantes": encoder_svg_local("include/Nantes.svg"),
    "Marseille": encoder_svg_local("include/OM.svg"),
    "Lyon": encoder_svg_local("include/OL.svg"),
    "Lorient": encoder_svg_local("include/Lorient.svg"),
    "Le Havre": encoder_svg_local("include/LeHavre.svg"),
    "Strasbourg": encoder_svg_local("include/Strasbourg.svg"),
    "Lille": encoder_svg_local("include/Lille.svg"),
    "Paris FC": "https://upload.wikimedia.org/wikipedia/fr/d/db/Logo_Paris_FC_2011.svg",
    "Toulouse":encoder_svg_local("include/Toulouse.svg")
    # ... et ainsi de suite
}

# 1. On récupère les données (une seule simulation comme avant)
import plotly.graph_objects as go

# 1. On prépare les données (comme avant)
liste_classements = simuler_saison_et_tracker_rangs(calendrier_25_26)

# 2. Création de la figure de base
fig = go.Figure()

# 3. Ajout des lignes pour chaque équipe (statiques au début)
for team in TEAMS:
    # On initialise avec la journée 1
    fig.add_trace(go.Scatter(
        x=[1], 
        y=[liste_classements[0].index.get_loc(team) + 1],
        mode='lines+markers+text',
        name=team,
        line=dict(width=2)
    ))

# 4. CRÉATION DES FRAMES (C'est ici que les logos bougent)
frames = []
for i in range(len(liste_classements)):
    frame_data = []
    current_layout_images = []
    
    for team in TEAMS:
        # Position de l'équipe à la journée i+1
        y_pos = liste_classements[i].index.get_loc(team) + 1
        x_pos = i + 1
        
        # Données de la ligne (historique jusqu'à la journée i)
        historique_x = list(range(1, i + 2))
        historique_y = [liste_classements[j].index.get_loc(team) + 1 for j in range(i + 1)]
        
        frame_data.append(go.Scatter(x=historique_x, y=historique_y))
        
        # AJOUT DU LOGO QUI BOUGE
        if team in LOGOS:
            current_layout_images.append(dict(
                source=LOGOS[team],
                xref="x", yref="y",
                x=x_pos, y=y_pos,
                sizex=2, sizey=2,
                xanchor="center", yanchor="middle",
                layer="above"
            ))
            
    frames.append(go.Frame(data=frame_data, layout=dict(images=current_layout_images), name=str(i+1)))

fig.frames = frames



# 5. CONFIGURATION DU LAYOUT ET DES BOUTONS
# 5. CONFIGURATION DU LAYOUT AVEC GLISSEMENT FLUIDE
fig.update_layout(
    title="Simulation Ligue 1 McDonald 2025-2026",
    title_x=0,
    x=-0.1,
    yaxis=dict(autorange="reversed", range=[18.5, 0.5], dtick=1, title="Rang"),
    xaxis=dict(range=[1, 35], dtick=1, title="Journée"),
    height=900,
    width=900,
    template="plotly_white",
    margin=dict(l=0, r=0, t=40, b=0),
    showlegend=False,
    

    updatemenus=[dict(
        type="buttons",
        showactive=False,
        x=-0.1, y=1.2,
        buttons=[dict(
            label="▶ Lancer la simulation",
            method="animate",
            args=[None, {
                "frame": {"duration": 2000, "redraw": True},
                "fromcurrent": True,
                "transition": {"duration": 1800, "easing": "cubic-in-out"}
            }]
        )]
    )]
)

# --- ZONES COLORÉES ---
fig.add_hrect(y0=0.5, y1=3.5, fillcolor="blue", opacity=0.08, 
              annotation_text="LIGUE DES CHAMPIONS", annotation_position="inside right")
fig.add_hrect(y0=17.5, y1=18.5, fillcolor="red", opacity=0.08, 
              annotation_text="ZONE DE RELÉGATION", annotation_position="inside right")
fig.add_hrect(y0=15.5, y1=17.5, fillcolor="pink", opacity=0.08, 
              annotation_text="ZONE DE BARAGE", annotation_position="inside right")
fig.add_hrect(y0=3.5, y1=5.5, fillcolor="violet", opacity=0.08, 
              annotation_text="LIGUE EUROPA", annotation_position="inside right")


st.plotly_chart(fig, use_container_width=True)
