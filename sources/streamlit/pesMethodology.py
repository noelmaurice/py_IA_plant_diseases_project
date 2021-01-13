import streamlit as st
import global_lib as gl

def main():
    gl.banner()
    
    st.markdown('# Méthodologie')
    st.markdown('## Approche')
    st.markdown("""Ce projet est un problème de classification d'images. Nous mettons en oeuvre des techniques de Deep Learning 
    en nous appuyant sur un réseau de neurones.  
    L'entraînement du modèle de segmentation est supervisé grâce au jeu de données **plantvillage-dataset**.  
    L'entraînement du modèle d'identification est supervisé grâce au jeu de données **new-plant-diseases-dataset**.  
    Les images de ces deux datasets sont organisées au sein de répertoires portant le nom des classes. Les datasets 
    respectent déjà le formalisme attendu par Keras.""")
    st.markdown("## Exploitation des images (modèle d'identification)")
    st.markdown("""
Avant d'exploiter les données, nous effectuons les étapes de preprocessing suivantes :  
  * Conversion RVB en BVR
  * Centrage sur 0 de chaque canal de couleur, sans mise à l'échelle.

Puis, nous procédons aussi à de l'augmentation de données pour accroître le nombre d'images utilisées pour l'entraînement 
du modèle. Nous utilisons les transformations suivantes :
  * Rotation
  * Décalage sur la hauteur
  * Décalage sur la largeur
  * Zoom
  * Retournement horizontal
  
Enfin, nous décidons d'ajouter une nouvelle classe 'Autres' au jeu de données initial pour permettre au modèle de détecter la 
présence d'objets qui ne sont pas des feuilles de plantes. La classe 'Autres' contient des images diverses (véhicules, animaux, ...)
provenant du jeu de données **Open Image Dataset**.""")

    gl.copyright()
