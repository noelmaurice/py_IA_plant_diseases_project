import streamlit as st
import pesPresentation
import pesDataset
import pesExploration
import pesMethodology
import pesModeling
import pesPrediction
import pesConclusion
import pesSegmentation 
import pesPCA 

def main():
    pages = {
        "Présentation"           : pesPresentation,
        "Jeux de données"        : pesDataset,
        "Exploration"            : pesExploration,
        "PCA"                    : pesPCA,
        "Méthodologie"           : pesMethodology,
        "Modélisation"           : pesModeling,
        "Segmentation (démo)"    : pesSegmentation,
        "Identification (démo)"  : pesPrediction,
        "Conclusion"             : pesConclusion
    }

    st.sidebar.markdown('## Projet pyStill')
    
    selection = st.sidebar.radio("Menu", list(pages.keys()))
    st.sidebar.info(
"""
Promotion Data Scientist Bootcamp Octobre 2020

__Participants :__  
[![LinkedIn](https://www.linkedin.com/favicon.ico)](https://www.linkedin.com/in/laxilais/) AXILAIS Loïc  
[![LinkedIn](https://www.linkedin.com/favicon.ico)](https://www.linkedin.com/in/noel-maurice-91208b78/) MAURICE Noël  
[![LinkedIn](https://www.linkedin.com/favicon.ico)](https://www.linkedin.com/in/mathis-ralaivao-8a3856131/) RALAIVAO Mathis  
"""
    )
    page = pages[selection]
    page.main()

main()
