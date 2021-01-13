import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import global_lib as gl


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.0f}%'.format(p=pct)
    return my_autopct

def get_last_segment(val):
    seg = val.split(os.sep)
    return seg[len(seg)-1]

def get_last_segment_pos(val, pos):
    last_seg = get_last_segment(val)
    if '___' in last_seg:
        return last_seg.split('___')[pos]
    else:
        return ''

def get_sample_type(val):
    seg = val.split(os.sep)
    sample_type = ['train', 'valid', 'test', 'color', 'grayscale', 'segmented']
    for st in sample_type:
        if st in seg:
            return st
    return ''

#@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('./data/filelist.csv', sep=';', encoding='ansi')
    return df

def paragrapheIntroduction():

    st.markdown("""Nous avons réalisé une analyse exploratoire des jeux de données sur Kaggle avec pour objectifs :
  * Comprendre la nature des données disponibles et leur organisation
  * Vérifier la qualité des données (existence de doublons, problème de complétude)""")

    st.markdown("""Après avoir téléchargé tous les jeux de données Kaggle, nous avons procédé ainsi :
  * Inventaire exhaustif des fichiers de chaque dataset
  * Analyse du nombre de fichiers par type de fichier et par dataset
  * Etude de la distribution des dimensions des fichiers images
  * Etude de la compatibilité de l'organisation des données avec les outils Keras
  * Sélection des données éligibles pour l'entraînement du modèle
  * Etude de la répartition de chaque classe dans les jeux de données retenus""")

def paragrapheInventaire():
    st.markdown("## Inventaire")
    with st.spinner(text="Préparation de la page ..."):
        # Lecture du fichier contenant l'inventaire des fichiers disponibles
        df =load_data()
    # Mise à jour de l'affichage
    st.markdown("Un extrait de l'inventaire des fichiers disponibles est présenté dans le dataframe suivant :")
    st.write(df.head(10))
    return df

#@st.cache(allow_output_mutation=True)
def loadTypeFichier(df):
    df['ext'] = df['ext'].apply(lambda x: x.upper())
    df = df.replace({'ext': {'JPEG': 'JPG'}})
    df_type = df.groupby(['dataset', 'ext'], as_index=False).agg({'file':'count'})
    # Création du dataframe dénombrant les fichiers de type image par dataset
    df_img = df_type[df_type['ext'].isin(['PNG', 'JPG'])].groupby(['dataset'], as_index=False).agg({'file':'sum'}).sort_values(by=['file'], ascending=False)
    # Création du camembert présentant le résultat
    fig = plt.figure(figsize=(5,5))
    df_img.plot.pie(y='file', title='Nombre de fichiers images par dataset', autopct=make_autopct(df_img['file']))
    fig = plt.figure(figsize=(5,5))
    plt.pie(x=df_img['file'], labels=df_img['dataset'], autopct='%1.1f%%')
    plt.title('Nombre de fichiers images par dataset')
    return df_type, fig

def paragrapheTypeFichier(df):
    st.markdown("## Types de fichier")
    with st.spinner(text="Préparation de la page ..."):
        # Création du dataframe dénombrant les fichiers par type de fichiers et par dataset
        df_type, fig = loadTypeFichier(df)
    # Mise à jour de l'affichage
    st.write(df_type)
    st.markdown("Les fichiers sont majoritairement des images (au format png ou jpg) :")
    st.pyplot(fig)

#@st.cache(allow_output_mutation=True)
def loadDistribution(df):
    def internalPlot(pos, dataset):
        ax = fig.add_subplot(pos)
        df_plt = df_dim[df_dim['dataset']==dataset]
        df_plt['size'] = df_plt['file'].apply(lambda x: x if x<500 else 500)
        ax.scatter(x=df_plt['width'], y=df_plt['height'], s=df_plt['size'])
        ax.set_title(dataset)
        ax.set_xlabel('width')
        ax.set_ylabel('height')

    # Calculer la répartition des tailles d'image (hauteur par largeur en fonction du dataset et de la couleur)
    df_dim = df[df.ext.isin(['JPG', 'PNG']) & ~df[['size', 'height', 'width']].isna().any(axis = 1)].groupby(['dataset', 'color', 'height', 'width'], as_index=False).agg({'file':'count'})
    # Afficher la répartition des tailles d'image
    fig = plt.figure(figsize=(20,10))
    internalPlot(231, 'plantclef-2019-amazon-rainforest-plants-images')
    internalPlot(232, 'v2-plant-seedlings-dataset')
    internalPlot(234, 'plantvillage-dataset')
    internalPlot(235, 'plant-disease')
    internalPlot(236, 'new-plant-diseases-dataset')
    return fig

def paragrapheDistribution(df):
    st.markdown("## Distribution")
    with st.spinner(text="Préparation de la page ..."):
        fig = loadDistribution(df)
    # Mise à jour de l'affichage
    st.markdown("Si l'on s'intéresse aux dimensions des images (hauteur x largeur), on constate des distributions hétérogènes.")
    st.pyplot(fig)
    st.markdown("""On peut écarter les datasets **plantclef-2019-amazon-rainforest-plants-images** et **v2-plant-seedlings-dataset** 
    qui contiennent peu de données et des distributions hétérogènes.""")

#@st.cache(allow_output_mutation=True)
def loadOrganisation(df):
    df_disease = df[df['dataset'].isin(['new-plant-diseases-dataset', 'plant-disease', 'plantvillage-dataset'])]
    df_disease.height = df_disease.height.astype(int)
    df_disease.width = df_disease.width.astype(int)
    df_disease.channel = df_disease.channel.astype(int)
    # Suppression des doublons dans le jeu de données plant-disease
    df_disease['doublon'] = df_disease[df_disease['dataset']=='plant-disease'].folder.apply(lambda x: True if x.split(os.sep)[1]=='dataset' else False)
    df_disease = df_disease.drop(df_disease[(df_disease['dataset']=='plant-disease') & (df_disease['doublon'])].index)
    df_disease.drop('doublon', axis=1, inplace=True)
    # Calcul des classes de données
    with open('./data/class_params.json', 'r') as f_in:
        plant_disease_dict = json.load(f_in)
    class_dict = {value['Indice']:key for (key,value) in plant_disease_dict.items()}
    df_disease['class'] = df_disease.folder.apply(lambda x: get_last_segment(x))
    df_disease['plant'] = df_disease.folder.apply(lambda x: get_last_segment_pos(x, 0))
    df_disease['disease'] = df_disease.folder.apply(lambda x: get_last_segment_pos(x, 1))
    df_disease['sample'] = df_disease.folder.apply(lambda x: get_sample_type(x))
    # Suppression des données de test du dataset new-plant-diseases-dataset
    df_disease=df_disease[~((df_disease['dataset']=='new-plant-diseases-dataset') & (df_disease['sample']=='test'))]
    # Récupération des libellés en français
    df_disease['plant_fr'] = df_disease['class'].apply(lambda x: plant_disease_dict[x]['Plant'])
    df_disease['disease_fr'] = df_disease['class'].apply(lambda x: plant_disease_dict[x]['Disease'])
    disease_count = df_disease.groupby(['dataset', 'sample', 'disease_fr'], as_index=False).file.count()
    # Les 33 images de l'échantillon de test du dataset new-plant-diseases-dataset n'ont pas de classe, elles sont ignorées
    disease_count = pd.pivot_table(disease_count[~((disease_count['dataset']=='new-plant-diseases-dataset') & (disease_count['sample']=='test'))], index='disease_fr', columns=['dataset', 'sample'], aggfunc='sum').reset_index()
    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(131)
    ax1.bar(range(len(disease_count)), disease_count[('file', 'plant-disease','train')], label='Train')
    ax1.bar(range(len(disease_count)), disease_count[('file', 'plant-disease','test')], bottom = disease_count[('file', 'plant-disease','train')], label='Test')
    ax1.set_xticks(disease_count.index)
    ax1.set_xticklabels(disease_count.disease_fr.values, rotation=90)
    ax1.set_xlabel('plant-disease')
    ax1.set_title("Nombre d'images par échantillon et par maladie")
    ax1.patch.set_facecolor('white')
    ax1.legend()

    ax2 = fig.add_subplot(132)
    ax2.bar(range(len(disease_count)), disease_count[('file', 'new-plant-diseases-dataset','train')], label='Train')
    ax2.bar(range(len(disease_count)), disease_count[('file', 'new-plant-diseases-dataset','valid')], bottom = disease_count[('file', 'new-plant-diseases-dataset','train')], label='Valid')
    ax2.set_xticks(disease_count.index)
    ax2.set_xticklabels(disease_count.disease_fr.values, rotation=90)
    ax2.set_xlabel('new-plant-diseases-dataset')
    ax2.set_title("Nombre d'images par échantillon et par maladie")
    ax2.patch.set_facecolor('white')
    ax2.legend()

    ax3 = fig.add_subplot(133)
    ax3.bar(range(len(disease_count)), disease_count[('file', 'plantvillage-dataset','color')], label='Couleur')
    ax3.bar(range(len(disease_count)), disease_count[('file', 'plantvillage-dataset','grayscale')], bottom = disease_count[('file', 'plantvillage-dataset','color')], label='Echelle de gris')
    ax3.bar(range(len(disease_count)), disease_count[('file', 'plantvillage-dataset','segmented')], bottom = disease_count[('file', 'plantvillage-dataset','color')] + disease_count[('file', 'plantvillage-dataset','grayscale')], label='Détourée')
    ax3.set_xticks(disease_count.index)
    ax3.set_xticklabels(disease_count.disease_fr.values, rotation=90)
    ax3.set_xlabel('plantvillage-dataset')
    ax3.set_title("Nombre d'images par échantillon et par maladie")
    ax3.patch.set_facecolor('white')
    ax3.legend()
    return df_disease, fig

def paragrapheOrganisation(df):
    st.markdown("## Organisation")
    with st.spinner(text="Préparation de la page ..."):
        df_disease, fig = loadOrganisation(df)

    # Mise à jour de l'affichage
    st.markdown("""A première vue, on pourrait penser que le dataset **plantvillage-dataset** est le plus intéressant car il 
    possède le plus grand nombre d'images. Mais en analysant d'un peu plus près l'organisation des données, on constate que les 
    données sont redondantes :""")
    st.pyplot(fig)
    st.markdown("On constate que le dataset **plantvillage-dataset** dispose en fait de 3 fois le même jeu d'informations : ")
    st.markdown("""
                * images en couleurs,  
                * images en échelle de gris,  
                * images détourées""")
    st.markdown("""Remarque : Sur le graphique, la classe 'Aucune' semble surreprésentée par rapport aux autres. C'est normal car 
    elle cumule les images de plantes sans maladie de chaque espèce de plante.""")
    return df_disease

def paragrapheSelection():
    st.markdown("## Sélection")
    st.markdown("""Les données des datasets **plant-disease** et **new-plant-diseases-dataset** sont réparties en deux échantillons 
    (train et test) alors que les données du dataset **plantvillage-dataset** se composent de trois échantillons (couleur, gris et 
    détourée). Finalement, c'est le dataset **new-plant-diseases-dataset** qui présente le plus d'images différentes, il semble être
    le plus intéressant.""")
    st.markdown("Quelques exemples d'images :")
    st.image('./images/sample.png', use_column_width=True)

#@st.cache(allow_output_mutation=True)
def loadRepartition(df_disease):
    plant_disease_count = df_disease[df_disease['dataset']=='new-plant-diseases-dataset'].groupby(['sample', 'plant_fr', 'disease_fr'], as_index=False).file.count()
    plant_disease_count = pd.pivot_table(plant_disease_count, index=['plant_fr', 'disease_fr'], columns=['sample'], aggfunc='sum').reset_index()
    plant_disease_count['total'] = plant_disease_count[('file', 'train')] + plant_disease_count[('file', 'valid')]
    plant_disease_count['order'] = plant_disease_count.disease_fr.apply(lambda x: 0 if x == 'Plante saine' else 1)
    plant_disease_count.sort_values(by=['plant_fr', 'order', 'disease_fr'], inplace=True)
    plant_disease_count.reset_index(inplace=True)
    plant_disease_count.drop('index', axis=1, inplace=True)
    fig = plt.figure(figsize=(20, 4))
    axe = fig.add_subplot(1, 1, 1)
    # On augmente artificiellement la dimension des ordonnées pour laisser de la place pour les noms des plantes
    ymax = plant_disease_count.total.max() + 2200
    plt.ylim([0, ymax])
    # On initialise les variables utilisées par la suite
    previous = plant_disease_count.iloc[0, 0] # On mémorise la plante précédente dans l'ordre d'apparition dans le dataset
    pos = 0 # On mémorise la position de l'étiquette d'abscisse à traiter
    start = -1 # On mémorise l'abscisse de démarrage de la première accolade
    yline = plant_disease_count.total.max() + 100 # On mémorise l'ordonnée des accolades
    xoffset = 0.2 # On gère un décroché sur l'axe des abscisses pour l'accolade
    yoffset = 50 # On gère un décroché sur l'axe des ordonnées pour l'accolade
    Xtick = [] # On initialise la liste permettant de gérer la position des étiquettes en abscisse
    Xlabel = [] # On initialise la liste permettant de gérer le libellé des étiquettes en abscisse
    XSpecial = [] # On initialise la liste permettant de gérer la position des étiquettes spéciales en abscisse (plante saine)
    label1 = '' # On initialise la légende de la barre des données d'entrainement
    label2 = '' # On initialise la légende de la barre des données de validation
    # Principe : on parcourt le dataset, lorsqu'on change de plante : 
    #   - On crée un espace entre groupe 
    #   - On affiche une accolade au dessus des barres avec le nom de la plante
    for i in range(len(plant_disease_count)):
        plant, disease, train, valid, total, order = plant_disease_count.iloc[i, :]
        if plant != previous: # Détection d'un changement de plante
            # Création de l'accolade
            plt.plot([start + xoffset, start + xoffset, pos - xoffset, pos - xoffset], [yline - yoffset, yline, yline, yline - yoffset], color='black')
            # Affichage du nom de la plante
            plt.text(start + (pos-start)//2, yline + 200, previous, rotation=90, backgroundcolor='white')
            previous = plant
            start = pos
            pos+=1
        # Gestion des étiquettes en abscisse
        Xtick.append(pos) # On mémorise la position de l'étiquette
        Xlabel.append(disease) # On mémorise le libellé de l'étiquette
        if order == 0:
            XSpecial.append(i) # On mémorise les étiquettes 'Plante saine'
        # Pour la dernière maladie, on affiche une légende
        if i == len(plant_disease_count)-1:
            label1 = 'train'
            label2 = 'valid'
        plt.bar([pos],[train], color=['blue'], label=label1)
        plt.bar([pos],[valid], bottom=[train], color=['orange'], label=label2)
        pos +=1
    # Gestion de la dernière accolade
    plt.plot([start + xoffset, start + xoffset, pos - xoffset, pos - xoffset], [yline - yoffset, yline, yline, yline - yoffset],color='black')
    plt.text(start + (pos-start)//2, yline + 200, previous, rotation=90, backgroundcolor='white')
    # Gestion de l'affichage des étiquettes en abscisse
    plt.xticks(Xtick, Xlabel, rotation=90)
    for i in XSpecial:
        plt.gca().get_xticklabels()[i].set_color("green")
    plt.legend()
    axe.patch.set_facecolor('white')
    plt.grid(True, linestyle = '--', axis='y', color='black', alpha=0.5)
    plt.title("Nombre d'images par échantillon par plante et par maladie pour le jeu de données new-plant-diseases-dataset")
    return fig

def paragrapheRepartition(df_disease):
    st.markdown("## Répartition au sein de new-plant-diseases-dataset")
    with st.spinner(text="Préparation de la page ..."):
        fig=loadRepartition(df_disease)
    # Mise à jour de l'affichage
    st.pyplot(fig)
    st.info("""La répartition des images entre classes est plutôt bien équilibrée, cela confirme notre choix d'utiliser ce
    dataset pour l'entraînement du modèle.""")


def main():
    gl.banner()
    st.markdown('# Exploration')

    paragrapheIntroduction()
    df = paragrapheInventaire()
    paragrapheTypeFichier(df)
    paragrapheDistribution(df)
    df2 = paragrapheOrganisation(df)
    paragrapheSelection()
    paragrapheRepartition(df2)

    gl.copyright()