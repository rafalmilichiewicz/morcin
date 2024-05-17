import streamlit as st
import requests
from PIL import Image
from io import BytesIO

def discover_dog():
    # Fetch all dogs from the database microservice
    response = requests.get('http://dogs_microservice:5000/dogs')
    dogs = response.json()

    # Lista do przechowywania tytułów psów
    dog_names = [dog['breed'] for dog in dogs]

    # Wybór psa z listy rozwijanej
    selected_dog_name = st.selectbox("Select a dog", dog_names)

    # Pobranie informacji o wybranym psie
    selected_dog = next(dog for dog in dogs if dog['breed'] == selected_dog_name)

    # Decode image from hex
    image_data = bytes.fromhex(selected_dog['image'])
    image = Image.open(BytesIO(image_data))

    # Wyświetlenie obrazka
    st.image(image, width=300)  # Zmniejszenie szerokości obrazka
    # Wyświetlenie danych
    st.header("Dog Information")
    st.markdown("---")  # Separator
    st.write("**Breed:**", selected_dog['breed'])
    st.write("**Curiosity:**", selected_dog['curiosity'])

if __name__ == "__main__":
    st.title("Discover a Dog")
    discover_dog()
