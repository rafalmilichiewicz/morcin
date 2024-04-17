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

    # Wykonanie zapytania o wszystkie psy
    select_query = "SELECT * FROM dogs"
    cursor.execute(select_query)
    results = cursor.fetchall()

    # Lista do przechowywania tytułów psów
    dog_names = [result[1] for result in results]

    # Wybór psa z listy rozwijanej
    selected_dog_name = st.selectbox("Select a dog", dog_names)

    # Pobranie informacji o wybranym psie
    selected_dog = [result for result in results if result[1] == selected_dog_name][0]

    # Wyświetlenie obrazka
    st.image(selected_dog[2], width=300)  # Zmniejszenie szerokości obrazka
    # Wyświetlenie danych
    st.header("Dog Information")
    st.markdown("---")  # Separator
    st.write("**Breed:**", selected_dog[1])
    st.write("**Curiosity:**", selected_dog[3])

    # Zamknięcie połączenia
    conn.close()

if __name__ == "__main__":
    st.title("Discover a Dog")
    discover_dog()
