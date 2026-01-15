import pandas as pd
import numpy as np
from scipy.stats import poisson
from matplotlib import pyplot as plt
import streamlit as st

import base64


#Calendrier
calendrier_25_26=pd.read_csv(f'data/calendrier_25_26.csv', encoding='utf-8-sig')

calendrier_25_26=calendrier_25_26[['wk','HomeTeam','AwayTeam','Date']]

# 1. Dictionnaire de traduction des mois
mois_fr = {
    'janvier': 'January', 'fevrier': 'February', 'mars': 'March', 'avril': 'April',
    'mai': 'May', 'juin': 'June', 'juillet': 'July', 'août': 'August', 'aout': 'August',
    'septembre': 'September', 'octobre': 'October', 'novembre': 'November', 'decembre': 'December',
    'décembre': 'December'
}

def formater_date_calendrier(date_str):
    if not isinstance(date_str, str): return date_str
    
    # Nettoyage (minuscule et suppression accents)
    s = date_str.lower().replace('û', 'u').replace('é', 'e')
    parts = s.split() # ["dimanche", "17", "aout", "2025"]
    
    if len(parts) >= 4:
        jour = parts[1]
        mois = mois_fr.get(parts[2], parts[2])
        annee = parts[3]
        # On crée une date temporaire pour la conversion
        temp_date = pd.to_datetime(f"{jour} {mois} {annee}", format='%d %B %Y')
        # On retourne le format final voulu JJ/MM/AAAA
        return temp_date.strftime('%d/%m/%Y')
    return date_str

# 2. Application au calendrier
# On remplace la colonne Date par le format JJ/MM/AAAA
calendrier_25_26['Date'] = calendrier_25_26['Date'].apply(formater_date_calendrier)

# 3. IMPORTANT : Pour que tes calculs de simulation fonctionnent,
# il faut quand même une version au format "datetime" interne
calendrier_25_26['Date_DT'] = pd.to_datetime(calendrier_25_26['Date'], format='%d/%m/%Y')

# À placer juste après le chargement du calendrier

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
    df = pd.read_csv(f'data/L1_{16+i}_{17+i}.csv', encoding='utf-8-sig')
    df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG','Date']].rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals','Date':'Date'})
    tous_les_matchs.append(df)

#Concaténation des datas
data_historique = pd.concat(tous_les_matchs, ignore_index=True)
data_historique.columns = data_historique.columns.str.strip()

data_historique['Date'] = pd.to_datetime(data_historique['Date'], dayfirst=True)
ref_date = data_historique['Date'].max()
data_historique['Days_Ago'] = (ref_date - data_historique['Date']).dt.days
data_historique['Weight'] = np.exp(-0.002 * data_historique['Days_Ago'])

HOME_GOALS_MEAN_GLOBAL = ((data_historique['HomeGoals']*data_historique['Weight']).sum())/(data_historique['Weight'].sum())
AWAY_GOALS_MEAN_GLOBAL = ((data_historique['AwayGoals']*data_historique['Weight']).sum())/(data_historique['Weight'].sum())

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

        Home_goals_mean_team=((home_matches['HomeGoals']*home_matches['Weight']).sum())/(home_matches['Weight'].sum())
        Away_goals_mean_team=((away_matches['AwayGoals']*away_matches['Weight']).sum())/(away_matches['Weight'].sum())
        Home_taken_mean_team=((home_matches['AwayGoals']*home_matches['Weight']).sum())/(home_matches['Weight'].sum())
        Away_taken_mean_team=((away_matches['HomeGoals']*away_matches['Weight']).sum())/(away_matches['Weight'].sum())
        
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

print(Capacity)
print(home_matches)

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


#Mettre le classement dans le bonne ordre
def afficher_classement_final(classement):
    
    
    criteres_de_tri = ['Pts', 'Diff', 'BP']
    
    ordre_decroissant = [False, False, False]
    
    classement_final = classement.sort_values(
        by=criteres_de_tri,
        ascending=ordre_decroissant
    )
    
    return classement_final

def recalculate_capacity(data_hist):
    # On recalcule les moyennes globales basées sur la base de données mise à jour
    h_glob = (data_hist['HomeGoals'] * data_hist['Weight']).sum() / data_hist['Weight'].sum()
    a_glob = (data_hist['AwayGoals'] * data_hist['Weight']).sum() / data_hist['Weight'].sum()

    for team in TEAMS:
        home_m = data_hist[data_hist['HomeTeam'] == team]
        away_m = data_hist[data_hist['AwayTeam'] == team]

        # Sécurité pour les promus ou équipes sans data récente
        if home_m['Weight'].sum() < 1 or away_m['Weight'].sum() < 1:
            promotion_L1(team)
        else:
            w_h, w_a = home_m['Weight'].sum(), away_m['Weight'].sum()
            
            # Moyennes pondérées de l'équipe
            h_g_m = (home_m['HomeGoals'] * home_m['Weight']).sum() / w_h
            a_g_m = (away_m['AwayGoals'] * away_m['Weight']).sum() / w_a
            h_t_m = (home_m['AwayGoals'] * home_m['Weight']).sum() / w_h
            a_t_m = (away_m['HomeGoals'] * away_m['Weight']).sum() / w_a
            
            Capacity[team] = {
                'Home_goals_capacity': h_g_m / h_glob,
                'Away_goals_capacity': a_g_m / a_glob,           
                'Home_taken_capacity': h_t_m / a_glob,         
                'Away_taken_capacity': a_t_m / h_glob,             
            }
    


def mettre_a_jour_classement_direct(classement, equipe_dom, equipe_ext, b_dom, b_ext):
    classement.loc[equipe_dom, 'Joués'] += 1
    classement.loc[equipe_ext, 'Joués'] += 1
    classement.loc[equipe_dom, 'BP'] += b_dom
    classement.loc[equipe_dom, 'BC'] += b_ext
    classement.loc[equipe_ext, 'BP'] += b_ext
    classement.loc[equipe_ext, 'BC'] += b_dom
    
    if b_dom > b_ext:
        classement.loc[equipe_dom, 'Pts'] += 3
        classement.loc[equipe_dom, 'G'] += 1
        classement.loc[equipe_ext, 'P'] += 1
    elif b_dom < b_ext:
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

print(calendrier_25_26)
#Simulation d'une journée
def simuler_wk(j, classement, calendrier, data_hist):
    matchs_journee = calendrier[calendrier['wk'] == j]
    nouveaux_matchs = []

    for index, match in matchs_journee.iterrows():
        e_dom, e_ext = match['HomeTeam'], match['AwayTeam']
        
        # Simulation unique du score
        b_dom, b_ext = simuler_match_poisson(e_dom, e_ext)
        
        # Mise à jour du classement
        classement = mettre_a_jour_classement_direct(classement, e_dom, e_ext, b_dom, b_ext)
        
        nouveaux_matchs.append({
            'HomeTeam': e_dom, 'AwayTeam': e_ext,
            'HomeGoals': b_dom, 'AwayGoals': b_ext,
            'Date': match['Date_DT']
        })

    # Mise à jour de la base historique pour la journée suivante
    df_nouveaux = pd.DataFrame(nouveaux_matchs)
    data_hist = pd.concat([data_hist, df_nouveaux], ignore_index=True)
    
    # Recalcul des poids et des capacités
    data_hist['Days_Ago'] = (data_hist['Date'].max() - data_hist['Date']).dt.days
    data_hist['Weight'] = np.exp(-0.002 * data_hist['Days_Ago'])
    recalculate_capacity(data_hist) 
    
    return classement, data_hist
#Simulation d'une saison
def simuler_saison_25_26():
    for teamA in TEAMS:
        for teamB in TEAMS:
            mettre_a_jour_classement_direct(classement,teamA,teamB)
    print(afficher_classement_final(classement))



#Obtention des rangs par equipes aprés chaque journée
def simuler_saison_et_tracker_rangs(calendrier_df):
    print("Simulation de la saison en cours ...")
    
    # On fait une copie locale de la base historique pour ne pas modifier 
    # la base globale à chaque rafraîchissement de la page Streamlit
    data_hist_temp = data_historique.copy()

    classement = {
        'Equipe': TEAMS,
        'Pts': 0, 'Joués': 0, 'G': 0, 'N': 0, 'P': 0, 'BP': 0, 'BC': 0, 'Diff': 0           
    }
    classement = pd.DataFrame(classement).set_index('Equipe')
    
    liste_classements = []

    for j in range(1, 35):
        # AJOUT de data_hist_temp dans les arguments
        # RÉCUPÉRATION des deux variables renvoyées
        classement_mis_a_jour_direct, data_hist_temp = simuler_wk(j, classement, calendrier_df, data_hist_temp)
        
        # On trie le classement avant de l'ajouter à la liste pour que les rangs soient bons
        classement_trie = afficher_classement_final(classement_mis_a_jour_direct)
        liste_classements.append(classement_trie.copy())
        
        # On repart du classement actuel pour la journée suivante
        classement = classement_mis_a_jour_direct.copy() 
        
    return liste_classements



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



st.divider()
st.header("⚽ Simulateur de Match Unique")

col1, col2 = st.columns(2)
with col1:
    equipe_a = st.selectbox("Équipe Domicile", TEAMS, key="home_sim")
with col2:
    equipe_b = st.selectbox("Équipe Extérieur", TEAMS, key="away_sim")

if st.button("Simuler le match"):
    if equipe_a == equipe_b:
        st.warning("Veuillez choisir deux équipes différentes.")
    else:
        
        # 2. Simulation du score
        resultat=simuler_match_poisson(equipe_a,equipe_b)
        b_a = resultat[0]
        b_b = resultat[1]

        # 3. Affichage Propre (Sans f-string complexe pour éviter les erreurs d'accolades)
        logo_a = LOGOS.get(equipe_a, "")
        logo_b = LOGOS.get(equipe_b, "")

        html_score = f"""
        <div style="text-align: center; border: 2px solid #e6e9ef; padding: 20px; border-radius: 15px; background-color: #ffffff; color: #31333F;">
            <div style="display: flex; justify-content: space-around; align-items: center;">
                <div style="flex: 1;">
                    <img src="{logo_a}" width="80" style="margin-bottom: 10px;"><br>
                    <span style="font-weight: bold; font-size: 1.2em;">{equipe_a}</span>
                </div>
                <div style="flex: 1; font-size: 3em; font-weight: 800; letter-spacing: 5px;">
                    {b_a} - {b_b}
                </div>
                <div style="flex: 1;">
                    <img src="{logo_b}" width="80" style="margin-bottom: 10px;"><br>
                    <span style="font-weight: bold; font-size: 1.2em;">{equipe_b}</span>
                </div>
            </div>
        </div>
        """
        st.markdown(html_score, unsafe_allow_html=True)
        
        # Petit feedback technique

#tracer_evolution_classement(creer_historique_par_club(simuler_saison_et_tracker_rangs(calendrier_25_26),TEAMS))

import plotly.express as px





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
    title_x=0.1,
    yaxis=dict(autorange="reversed", range=[18, 1], dtick=1, title="Rang"),
    xaxis=dict(range=[1, 35], dtick=1, title="Journée",domain=[0, 1]),
    height=800,
    width=800,
    template="plotly_white",
    showlegend=False,
    margin=dict(l=10, r=0, t=100, b=0),
    

    updatemenus=[dict(
        type="buttons",
        showactive=False,
        x=0, y=1.2,
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
st.divider()
st.header("⚽ Simulation en temps réel")
st.plotly_chart(fig, use_container_width=False)

