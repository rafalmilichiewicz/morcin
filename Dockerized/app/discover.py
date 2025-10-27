import streamlit as st
import mysql.connector
from PIL import Image
from io import BytesIO
from db.data import example_dogs
from db.env import get_db_connection
def discover_dog():
    # Open a new connection
    conn = get_db_connection()
    cursor = conn.cursor()
    # Execute query to fetch all dogs
    select_query = "SELECT * FROM dogs"
    cursor.execute(select_query)
    results = cursor.fetchall()
    if not results:
        if st.button("Dodaj przykładowe psiaki"):
            example_dogs()
            st.rerun()
    else:


        # List to store dog names
        dog_names = [result[1] for result in results]

        # Select a dog from dropdown
        selected_dog_name = st.selectbox("Select a dog", dog_names)

        # Fetch information about the selected dog
        selected_dog = [result for result in results if result[1] == selected_dog_name][0]

        # Display the image
        st.image(selected_dog[2], width=300)  # Resize image width
    
        # Display dog information
        st.header("Dog Information")
        st.markdown("---")  # Separator
        st.write("**Breed:**", selected_dog[1])
        st.write("**Curiosity:**", selected_dog[3])

    # Close connection
    conn.close()
