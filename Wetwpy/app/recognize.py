import streamlit as st
import model.model as ml

def recognize_dog():
    uploaded_image = st.file_uploader("Wybierz obraz do rozpoznania", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption='Wybrany obraz', use_column_width=True)
        
        # Use unique key for the button
        recognize_button = st.button("Rozpoznaj psiaka", key="button1")
        if recognize_button:
            with st.spinner("Rozpoznawanie psa..."):
                # Save the uploaded image to a temporary file
                temp_file_path = 'temp_image.jpg'
                with open(temp_file_path, 'wb') as f:
                    f.write(uploaded_image.read())
                # Call predict_breed function from the imported model module with the image path
                predictions = ml.predict_breed(temp_file_path)
            st.write("Wyniki rozpoznania:")
            for prediction in predictions:
                st.write(prediction)

if __name__ == "__main__":
    recognize_dog()
