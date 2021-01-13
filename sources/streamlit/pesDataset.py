import streamlit as st
import global_lib as gl

def main():
    gl.banner()
    
    st.markdown('# Jeux de données')
    
    st.markdown("""La première étape du projet consiste à trouver des jeux de données permettant d'entraîner notre modèle de 
    prédiction.  
    Nous avons étudié les datasets suivants :
    """)
    st.markdown("## COCO Common Objects in Context")
    st.markdown("__source :__ https://cocodataset.org/#home")
    st.markdown(""" Le site propose un ensemble de données labellisées (véhicules, animaux, mobiliers, …) mais il n’y a rien 
    d’utilisable sur les plantes (à part quelques plantes en pot).  
    Par contre, le site présente le principe de détection d'objet et de segmentation dont nous nous sommes inspirés.  
    Par ailleurs, Le site est intéressant pour tester notre modèle sur des photos ne représentant pas des plantes afin de vérifier 
    comment il se comporte.""")
    st.markdown("## Open Image Dataset")
    st.markdown("__source :__ https://storage.googleapis.com/openimages/web/download.html")
    st.markdown("""Le site propose un vaste ensemble de données labellisées (près de 20 000 classes) ainsi que des jeux de 
    données avec Bounded Box et masque de segmentation (600 classes). Comme le COCO Dataset, il n'y a rien d'exploitable concernant 
    les maladies des plantes.""")
    st.markdown("## Kaggle")
    st.markdown("### Identification d’espèces")
    st.markdown("""
                * __source :__ https://www.kaggle.com/vbookshelf/v2-plant-seedlings-dataset  
                * __source :__ https://www.kaggle.com/miljan/plantclef-2019-amazon-rainforest-plants-images""")
    st.markdown("### Reconnaissance de maladies")
    st.markdown("""
                * __source :__ https://www.kaggle.com/vipoooool/new-plant-diseases-dataset  
                * __source :__ https://www.kaggle.com/saroz014/plant-disease  
                * __source :__ https://www.kaggle.com/abdallahalidev/plantvillage-dataset""")
    st.markdown("""Les jeux de données sur Kaggle semblent particulièrement adaptés à notre problématique.  
    Nous nous sommes concentrés dessus à travers l'exploration présentée à la page suivante.""")
    st.markdown("## Plant Village")
    st.markdown("""__source :__ https://plantvillage.psu.edu""")
    st.markdown("""Le site dispose d'une base de connaissances sur les maladies des plantes que nous pouvons valoriser une fois la
    reconnaissance par prédiction de la plante et de la maladie effectuée.""")

    gl.copyright()