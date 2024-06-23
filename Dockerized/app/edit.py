import streamlit as st
import mysql.connector
from PIL import Image
from io import BytesIO

def edit_dog():
    st.title("Dodaj/usuń dane psiaka")

    # Opcje dodawania/usuwania psa
    add_or_remove = st.radio("Co chcesz zrobić?", ("Dodaj psa", "Usuń psa"))

    if add_or_remove == "Dodaj psa":
        # Formularz dodawania psa
        st.header("Dodaj nowego psa")
        breed = st.text_input("Rasa psa:")
        curiosity = st.text_input("Ciekawostki o psie:")
        image_upload = st.file_uploader("Wybierz obraz psa:", type=["jpg", "jpeg", "png"])

        if st.button("Dodaj psa"):
            if breed and curiosity and image_upload:
                # Konwersja obrazka do formatu do zapisu w bazie danych
                image = Image.open(image_upload)
                image_bytes = BytesIO()
                image.save(image_bytes, format=image.format)
                image_data = image_bytes.getvalue()

                # Dodanie psa do bazy danych
                conn = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="password",  # Ustaw swoje hasło
                    database="dogs_db"
                )
                cursor = conn.cursor()
                insert_query = "INSERT INTO dogs (breed, image, curiosity) VALUES (%s, %s, %s)"
                cursor.execute(insert_query, (breed, image_data, curiosity))
                conn.commit()
                conn.close()

                st.success("Pies został pomyślnie dodany do bazy danych!")
            else:
                st.error("Wypełnij wszystkie pola!")

    elif add_or_remove == "Usuń psa":
        # Formularz usuwania psa
        st.header("Usuń psa")
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="password",  # Ustaw swoje hasło
            database="dogs_db"
        )
        cursor = conn.cursor()
        select_query = "SELECT * FROM dogs"
        cursor.execute(select_query)
        results = cursor.fetchall()
        dog_names = [result[1] for result in results]
        selected_dog_name = st.selectbox("Wybierz psa do usunięcia:", dog_names)

        if st.button("Usuń psa"):
            delete_query = "DELETE FROM dogs WHERE breed = %s"
            cursor.execute(delete_query, (selected_dog_name,))
            conn.commit()
            conn.close()
            st.success(f"Pies {selected_dog_name} został pomyślnie usunięty z bazy danych!")

if __name__ == "__main__":
    edit_dog()