import streamlit as st
import global_lib as gl

def main():
    gl.banner()
    
    st.markdown('# Conclusion')
    st.markdown('## Bilan')
    st.markdown("""
    Le projet **pyStill** nous a permis de mettre en pratique nos connaissances acquises durant le cursus 'Data Scientist'.  
    Nous avons pu :  
      * Réaliser le projet selon une démarche scientifique : Hypothèse, Expérimentation, Validation (ou non),  
      * Elaborer différents modèles, les entraîner et démontrer leur pertinence et leur qualité à travers ce site,  
      * Nous confronter aux difficultés de mise en oeuvre d'un projet en équipe et en distanciel.
    """)
    st.markdown('## Perspectives')
    st.markdown("""
    Nous avons répondu à la demande initiale de détection de plante et de maladie éventuelle à partir d'une photographie.  
    Ce projet pourrait être avantageusement complété de fonctionnalités supplémentaires :  
      * **Robustesse du modèle d'identification :**  
    Nous avons constaté que lorsque le modèle reçoit une image d’une plante pour laquelle il n’a pas été entrainé, il 
    identifie cette plante comme étant connue avec un indice de confiance relativement élevé.  
    De la même façon que nous avons entraîné notre modèle à reconnaître une image qui n'est pas une plante avec la classe 'Autres',
    il faudrait l'entraîner à reconnaître une 'Autre Plante' que celles dont la classification est connue.  
      * **Domaine de prédiction :**  
    Nos modèles de prédiction présentent de bonnes performances mais leur faiblesse principale est le nombre de classes prédites 
    (38 classes de couple plante-maladie et une classe 'Autres').  
    Pour une application professionnelle utilisable dans le monde agricole, il faudrait augmenter le nombre de classes prédites 
    de manière significative (il existe environ 7000 variétés végétales cultivées à travers le monde). 
      * **Gamme de services :**  
    L’identification et la classification des plantes et de leurs maladies respectives est une étape nécessaire mais insuffisante 
    pour répondre aux enjeux culturaux.  
    Il faudrait disposer d'informations sur l'état de santé générale de la plante (et pas seulement sur une feuille) permettant de 
    proposer un traitement curatif ou préventif et un dosage de produits phytosanitaires pour endiguer la maladie.  
    Le système pourrait aussi donner des indications sur le coût de traitement à l’hectare (coût du travail, coût des produits, ...) 
    permettant ainsi aux producteurs de vérifier la rentabilité économique du traitement.  
    """)

    gl.copyright()