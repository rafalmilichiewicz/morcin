import streamlit as st
import mysql.connector
from PIL import Image
from io import BytesIO

def discover_dog():
    # Otwarcie nowego połączenia
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",  # Ustaw swoje hasło
        database="dogs_db"
    )
    cursor = conn.cursor()

    # Wykonanie zapytania o wyświetlenie danych
    select_query = "SELECT * FROM dogs LIMIT 1"
    cursor.execute(select_query)
    result = cursor.fetchone()
    # Wyświetlenie obrazka
    st.image(result[2], width=300)  # Zmniejszenie szerokości obrazka
    # Wyświetlenie danych
    st.header("Dog Information")
    st.markdown("---")  # Separator
    st.write("**Breed:**", result[1])
    st.write("**Curiosity:**", result[3])

    # Zamknięcie połączenia
    conn.close()

if __name__ == "__main__":
    st.title("Discover a Dog")
    discover_dog()
