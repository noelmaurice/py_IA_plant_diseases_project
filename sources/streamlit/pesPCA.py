import streamlit as st
import global_lib as gl

def main():
    gl.banner()
    
    st.markdown('# Principal Component Analysis')
    st.markdown('## Objectif')
    st.markdown("""Avant de nous lancer dans les algorithmes de Deep Learning, nous avons étudié la possibilité de réaliser des 
    prédictions à partir d'un modèle de Machine Learning (comme le RandomForrest par exemple).  
    Cependant nous étions confrontés au grand nombre de dimensions. En effet, les images du dataset **plantvillage-dataset** ont une 
    taille de 256 pixels de largeur x 256 pixels de hauteur x 3 canaux (vert, rouge, bleu), soit 196 608 dimensions.  
    Nous avons donc cherché à réduire le nombre de dimensions en utilisant la méthode PCA.""")
    st.markdown('## Résultats')
    st.markdown("Nous avons appliqué une méthode PCA sur quelques classes en conservant 95% de la variance.")
    st.image('./images/PCA1.png', use_column_width=True)
    st.markdown("Nous arrivons en moyenne à réduire le nombre de dimensions de 99.486%.")
    st.markdown("""Cette méthode permet de diminuer grandement la dimension des données, avec une réduction de plus de 99% pour une 
    perte de seulement 5% de la variance. Après réduction, il reste un peu plus de 1000 dimensions.""")
    st.markdown("""Nous avons approfondi l'analyse en étudiant des graphiques bidimensionnels prenant en compte les deux premiers
    composants des images réduites par PCA. Ils permettent souvent d'expliquer la majeure partie de la variance des données. Nous 
    pouvons visualiser la répartition des images par classe, en affichant certaines images pour avoir une meilleure compréhension 
    de cette répartiton.""")
    st.markdown("Voici quelques exemples sur les pommes :")
    st.image('./images/PCA_AppleScab.png', use_column_width=True)
    st.markdown("On constate sur ce premier graphique que la rotation des images influence le classement.")
    st.image('./images/PCA_AppleBlackRot.png', use_column_width=True)
    st.markdown("On constate sur ce second graphique que le premier groupe d'images est ordonné horizontalement selon la limunosité.")
    st.image('./images/PCA_AppleCedarAppleRust.png', use_column_width=True)
    st.markdown("L'impact de la luminosité sur la classification est encore plus flagrant sur ce dernier graphique.")
    st.markdown('## Exploitation')
    st.markdown("""Malgré une réduction importante des dimensions, les algorithmes de Machine Learning n'ont pas donné de résultats
    satisfaisants en matière de prédiction. Nous avons donc décidé de nous tourner vers des algorithmes de Deep Learning.""")

    gl.copyright()