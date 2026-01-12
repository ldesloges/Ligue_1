import pandas as pd
import numpy as np
from scipy.stats import poisson
from matplotlib import pyplot as plt
import streamlit as st

import base64


calendrier_25_26=pd.read_csv(f'data/calendrierPL.csv')

calendrier_25_26=calendrier_25_26[['matchday','home_team','away_team']]

calendrier_25_26 = calendrier_25_26.rename(columns={'matchday': 'wk'})
calendrier_25_26 = calendrier_25_26.rename(columns={'home_team': 'HomeTeam'})
calendrier_25_26 = calendrier_25_26.rename(columns={'away_team': 'AwayTeam'})

print(calendrier_25_26)

def encoder_svg_local(chemin_fichier):
    with open(chemin_fichier, "rb") as f:
        image_data = f.read()
        base64_string = base64.b64encode(image_data).decode("utf-8")
        return f"data:image/svg+xml;base64,{base64_string}"

# Utilise directement l'URL de l'image que tu as hébergée
logo_url = "https://preview.redd.it/nouveau-logo-de-la-ligue-1-mcdonalds-pour-la-saison-2024-v0-pespi1nju2rc1.jpeg?width=1080&crop=smart&auto=webp&s=adc08fd2b07e1030b8ce301533d1678eb2d94d5c" 

st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 30px;">
        <img src="{logo_url}" width="100">
    </div>
    """,
    unsafe_allow_html=True
)


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


#Création de la DataFrame
tous_les_matchs = []
for i in range(9):
    df = pd.read_csv(f'data/PL_{16+i}_{17+i}.csv')
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

def simuler_monte_carlo(n_simulations=1000):
    # 1. Pré-calcul des capacités pour éviter de chercher dans le dictionnaire Capacity à chaque match
    # Format : { Equipe: (Capa_Buts_Dom, Capa_Buts_Ext, Capa_Pris_Dom, Capa_Pris_Ext) }
    capa_fast = {t: (
        Capacity[t]['Home_goals_capacity'], 
        Capacity[t]['Away_goals_capacity'],
        Capacity[t]['Home_taken_capacity'], 
        Capacity[t]['Away_taken_capacity']
    ) for t in TEAMS}

    # 2. Conversion du calendrier en liste de tuples (plus rapide que de lire le DataFrame)
    matchs_liste = list(calendrier_25_26[['HomeTeam', 'AwayTeam']].itertuples(index=False, name=None))
    
    resultats_positions = {team: [] for team in TEAMS}
    progress_bar = st.progress(0)

    for i in range(n_simulations):
        # On utilise un dictionnaire simple {Equipe: [Points, Diff, ButsPour]}
        # On ne touche pas au DataFrame Pandas ici pour gagner du temps
        scores = {team: [0, 0, 0] for team in TEAMS}

        for h_team, a_team in matchs_liste:
            # Récupération ultra-rapide des capacités
            c_h = capa_fast[h_team]
            c_a = capa_fast[a_team]

            # Calcul des lambdas
            l_dom = HOME_GOALS_MEAN_GLOBAL * c_h[0] * c_a[3]
            l_ext = AWAY_GOALS_MEAN_GLOBAL * c_a[1] * c_h[2]

            # Simulation des buts (NumPy génère les deux d'un coup)
            b_h, b_a = np.random.poisson([l_dom, l_ext])

            # Mise à jour des points
            if b_h > b_a:
                scores[h_team][0] += 3
            elif b_a > b_h:
                scores[a_team][0] += 3
            else:
                scores[h_team][0] += 1
                scores[a_team][0] += 1
            
            # Mise à jour Diff et BP
            scores[h_team][1] += (b_h - b_a)
            scores[h_team][2] += b_h
            scores[a_team][1] += (b_a - b_h)
            scores[a_team][2] += b_a

        # Tri du classement (Critères : Points, puis Diff, puis BP)
        # sorted() est extrêmement performant en Python
        classement_trie = sorted(scores.items(), key=lambda x: (x[1][0], x[1][1], x[1][2]), reverse=True)
        
        # Enregistrement des rangs
        for rang, (team, _) in enumerate(classement_trie, 1):
            resultats_positions[team].append(rang)
        
        # Mise à jour de la barre toutes les 10 simulations pour ne pas ralentir l'affichage
        if i % 10 == 0 or i == n_simulations - 1:
            progress_bar.progress((i + 1) / n_simulations)

    return resultats_positions


def calculer_stats_probabilites(resultats_positions, n_simulations):
    stats = []
    for team, rangs in resultats_positions.items():
        rangs = np.array(rangs)
        stats.append({
            'Equipe': team,
            'Champion (%)': (np.sum(rangs == 1) / n_simulations) * 100,
            'Top 3 (%)': (np.sum(rangs <= 3) / n_simulations) * 100,
            'Relégation (%)': (np.sum(rangs >= 17) / n_simulations) * 100,
            'Rang Moyen': np.mean(rangs)
        })
    
    return pd.DataFrame(stats).sort_values(by='Champion (%)', ascending=False).set_index('Equipe')


import plotly.express as px

def tracer_heatmap_probabilites(resultats_positions):
    # 1. On calcule le rang moyen pour trier les équipes verticalement
    # Cela permet d'avoir une belle diagonale sur le graphique
    rangs_moyens = {team: np.mean(ranks) for team, ranks in resultats_positions.items()}
    equipes_triees = sorted(TEAMS, key=lambda x: rangs_moyens[x])

    nb_equipes = len(TEAMS)
    matrix_data = []
    
    for team in equipes_triees:
        rangs = resultats_positions[team]
        # On calcule le % de chances pour chaque place de 1 à 18
        comptage = [(rangs.count(pos) / len(rangs)) * 100 for pos in range(1, nb_equipes + 1)]
        matrix_data.append(comptage)
    
    fig = px.imshow(
        matrix_data,
        labels=dict(x="Position Finale", y="Équipe", color="Probabilité (%)"),
        x=list(range(1, nb_equipes + 1)),
        y=equipes_triees,
        color_continuous_scale="Viridis",
        text_auto=".1f", 
        aspect="auto"
    )

    fig.update_layout(
        title="Où vont-ils finir ? (Probabilités par position)",
        xaxis_title="Rang au classement",
        yaxis_title="Équipe",
        xaxis=dict(dtick=1), # Affiche tous les numéros de 1 à 18
        height=700
    )
    
    return fig

st.title("🏆 Simulation Ligue 1 2025-2026")
st.divider()
st.header("🎲 Analyse Prédictive (Monte-Carlo)")

n_simu = st.slider("Nombre de simulations", min_value=10, max_value=1000, value=100)

if st.button("Lancer l'Analyse Statistique", key="bouton_stats_1"):
    with st.spinner('Calcul des probabilités en cours...'):
        resultats = simuler_monte_carlo(n_simu)
        df_stats = calculer_stats_probabilites(resultats, n_simu)
        
        st.subheader(f"Résultats basés sur {n_simu} saisons simulées")
        
        # Affichage avec mise en forme
        st.subheader(f"📊 Statistiques détaillées")
        st.dataframe(df_stats.style.format({
            'Champion (%)': '{:.1f}%',
            'Top 3 (%)': '{:.1f}%',
            'Relégation (%)': '{:.1f}%',
            'Rang Moyen': '{:.2f}'
        }).background_gradient(cmap='Blues', subset=['Champion (%)', 'Top 3 (%)'])
          .background_gradient(cmap='Reds', subset=['Relégation (%)']))
        
        # 2. Affichage de la Heatmap (Visualisation de l'incertitude)
        st.subheader("🔥 Distribution des probabilités de classement")
        fig_heatmap = tracer_heatmap_probabilites(resultats)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        

        # Petit récapitulatif textuel
        top_team = df_stats.index[0]
        st.info(f"💡 D'après les simulations, **{top_team}** a la plus forte probabilité de finir champion ({df_stats.loc[top_team, 'Champion (%)']:.1f}%).")




