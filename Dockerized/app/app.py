import streamlit as st
from ui import main as show_ui
from recognize import recognize_dog
from discover import discover_dog
from edit import edit_dog
import mysql.connector

def main():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",  # Replace with your actual password
    )
    mycursor = conn.cursor()

    # Execute each SQL statement separately
    mycursor.execute("CREATE DATABASE IF NOT EXISTS dogs_db;")
    mycursor.execute("USE dogs_db;")
    mycursor.execute("""
    CREATE TABLE IF NOT EXISTS dogs (
        id INT AUTO_INCREMENT PRIMARY KEY,
        breed VARCHAR(100) NOT NULL,
        image LONGBLOB NOT NULL,
        curiosity TEXT NOT NULL
    );""")

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
