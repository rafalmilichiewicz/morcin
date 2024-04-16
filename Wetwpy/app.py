import streamlit as st
from ui import main as show_ui
from recognize import recognize_dog
from discover import discover_dog

def main():
    menu_choice = show_ui()
    
    if menu_choice == "Rozpoznaj psiaka":
        recognize_dog()
    elif menu_choice == "Odkryj psiaka":
        discover_dog()

if __name__ == "__main__":
    main()
