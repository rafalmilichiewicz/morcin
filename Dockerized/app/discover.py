import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import time

def discover_dog():
    retries = 5
    for attempt in range(retries):
        try:
            response = requests.get('http://dogs_microservice:5000/dogs')
            response.raise_for_status()
            dogs = response.json()

            dog_names = [dog['breed'] for dog in dogs]

            selected_dog_name = st.selectbox("Select a dog", dog_names)

            selected_dog = next((dog for dog in dogs if dog['breed'] == selected_dog_name), None)

            if selected_dog:
                image_data = bytes.fromhex(selected_dog['obrazek'])
                image = Image.open(BytesIO(image_data))

                st.image(image, width=300)
                st.header("Dog Information")
                st.markdown("---")
                st.write("**Breed:**", selected_dog['breed'])
                st.write("**Curiosity:**", selected_dog['curiosity'])
            else:
                st.error("Dog not found. Please select a valid dog.")
            break  # If successful, break out of retry loop
        except requests.RequestException as e:
            if attempt < retries - 1:
                st.warning(f"Error fetching data: {e}. Retrying...")
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                st.error(f"Max retries exceeded: {e}")

if __name__ == "__main__":
    st.title("Discover a Dog")
    discover_dog()
