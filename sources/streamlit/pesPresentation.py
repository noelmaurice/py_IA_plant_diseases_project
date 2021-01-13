import streamlit as st
import global_lib as gl

def main():
    gl.banner()
    
    st.markdown('# Présentation')

    col1, col2 = st.beta_columns([1, 3])

    with col1:
        st.image('./images/plantation.png', use_column_width=True)

    with col2:
        st.markdown("""
        La production végétale est à la base de toute la chaîne d'alimentation des humains et des animaux. La reconnaissance des 
        maladies sur les plantes est donc un enjeu majeur. C'est la première étape avant de pouvoir lancer un traitement sanitaire sur 
        une culture.

        Le projet __pyStill__ propose de détecter d'éventuelles maladies sur les plantes. Nous nous sommes concentrés sur les plantes
        suivantes : Cerise et griotte, Courge, Fraise, Framboise, Maïs, Myrtille, Orange, Pêche, Poivron, Pomme, Pomme de terre, 
        Raisin, Soja, Tomate.

        Le principe est simple, il suffit de fournir la photographie d'une feuille de la plante et le système indique si la plante est 
        saine. Dans le cas contraire, le système indique la maladie détectée.  

        Lorsqu'une maladie est détectée, le système propose quelques informations et conseils pour le traitement.  
        
        Ce projet s'appuie sur des algorithmes de Deep Learning entrainés sur quelques dizaines de milliers d'images.
        """)

    gl.copyright()
    