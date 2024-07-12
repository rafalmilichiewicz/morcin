import streamlit as st
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import model.ml as ml_model  # Rename the imported module to avoid conflict

# Enable GPU usage if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model and class indices
def load_model_and_indices():
    model_path = 'model/best_model.pth'
    class_indices_path = 'model/class_indices.npy'

    if not os.path.exists(model_path) or not os.path.exists(class_indices_path):
        st.error("Model files not found. Please train the model first.")
        return None, None

    num_classes = 120  # Assuming there are 120 classes
    model = ml_model.build_model(num_classes)  # Use the renamed module here
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with open(class_indices_path, 'rb') as f:
        class_indices = np.load(f, allow_pickle=True).item()

    return model, class_indices

# Function to predict the breed of a dog
def predict_breed(model, img_path, class_indices):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities, predicted_indices = torch.topk(torch.softmax(outputs, dim=1), 3)

    class_labels = {v: k for k, v in class_indices.items()}
    top3_predictions = [(class_labels[idx.item()], prob.item()) for idx, prob in zip(predicted_indices[0], probabilities[0])]

    return top3_predictions

# Streamlit app
def recognize_dog():
    st.title("Dog Breed Recognition")

    uploaded_image = st.file_uploader("Choose an image to recognize", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        recognize_button = st.button("Recognize Breed")
        if recognize_button:
            with st.spinner("Recognizing the dog..."):
                # Save the uploaded image to a temporary file
                temp_file_path = 'temp_image.jpg'
                with open(temp_file_path, 'wb') as f:
                    f.write(uploaded_image.read())
                
                # Load the model and class indices
                model, class_indices = load_model_and_indices()
                if model is None or class_indices is None:
                    return
                
                # Predict the breed
                predictions = predict_breed(model, temp_file_path, class_indices)
                
                st.write("Recognition Results:")
                for breed, prob in predictions:
                    st.write(f"Predicted breed: {breed}, Probability: {prob:.4f}")

if __name__ == "__main__":
    recognize_dog()
