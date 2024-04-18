import streamlit as st
from app.ui import main as show_ui
from app.recognize import recognize_dog
from app.discover import discover_dog
from app.edit import edit_dog

def main():
    # Initialize session state
    if "menu_choice" not in st.session_state:
        st.session_state.menu_choice = "Strona główna"
    
    menu_choice = show_ui()
    
    if menu_choice == "Rozpoznaj psiaka":
        recognize_dog()
    elif menu_choice == "Odkryj psiaka":
        discover_dog()
    elif menu_choice == "Dodaj/usuń dane psiaka":
        edit_dog()
    

if __name__ == "__main__":
    main()
