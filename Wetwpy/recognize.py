import streamlit as st
import model  # Importing the model module

def recognize_dog():
    uploaded_image = st.file_uploader("Wybierz obraz do rozpoznania", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption='Wybrany obraz', use_column_width=True)
        
        # Use unique key for the button
        recognize_button = st.button("Rozpoznaj psiaka", key="button1")
        if recognize_button:
            with st.spinner("Rozpoznawanie psa..."):
                img_path = 'image.jpg'  # Replace 'image.jpg' with the path to the dog image you want to classify
                predictions = model.predict_breed(img_path)  # Call predict_breed function from the imported model module with the image path
            st.write("Wyniki rozpoznania:")
            for prediction in predictions:
                st.write(prediction)

if __name__ == "__main__":
    recognize_dog()
