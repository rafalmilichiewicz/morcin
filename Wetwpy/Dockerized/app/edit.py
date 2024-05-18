import streamlit as st
import requests
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
                try:
                    # Konwersja obrazka do formatu do zapisu w bazie danych
                    image = Image.open(image_upload)
                    image_bytes = BytesIO()
                    image.save(image_bytes, format=image.format)
                    image_data = image_bytes.getvalue()

                    # Wysłanie żądania do mikroserwisu bazy danych
                    payload = {
                        'breed': breed,
                        'curiosity': curiosity,
                        'image': image_data.hex()  # Convert to hex string
                    }
                    response = requests.post('http://dogs_microservice:5000/dogs', json=payload)

                    if response.status_code == 200:
                        st.success("Pies został pomyślnie dodany do bazy danych!")
                    else:
                        st.error("Wystąpił błąd podczas dodawania psa do bazy danych.")
                except Exception as e:
                    st.error(f"Wystąpił błąd podczas dodawania psa: {e}")
            else:
                st.error("Wypełnij wszystkie pola!")

    elif add_or_remove == "Usuń psa":
        # Formularz usuwania psa
        st.header("Usuń psa")
        try:
            response = requests.get('http://dogs_microservice:5000/dogs')
            if response.status_code == 200:
                dogs = response.json()
                dog_names = [dog['breed'] for dog in dogs]
                selected_dog_name = st.selectbox("Wybierz psa do usunięcia:", dog_names)

                if st.button("Usuń psa"):
                    try:
                        response = requests.delete(f'http://dogs_microservice:5000/dogs/{selected_dog_name}')
                        if response.status_code == 200:
                            st.success(f"Pies {selected_dog_name} został pomyślnie usunięty z bazy danych!")
                        else:
                            st.error(f"Wystąpił błąd podczas usuwania psa {selected_dog_name} z bazy danych.")
                    except Exception as e:
                        st.error(f"Wystąpił błąd podczas usuwania psa: {e}")
            else:
                st.error("Wystąpił błąd podczas pobierania listy psów z bazy danych.")
        except Exception as e:
            st.error(f"Wystąpił błąd podczas komunikacji z mikroserwisem: {e}")

if __name__ == "__main__":
    edit_dog()
