# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import cv2
# import matplotlib.pyplot as plt

# # Load the SVM model and scaler correctly
# svm_model = joblib.load(r"E:/SVM_Model/SVM/Best_Model.pkl")  # Load SVM model
# scaler = joblib.load(r"E:/SVM_Model/SVM/scaler.pkl")  # Load StandardScaler


# # Streamlit UI
# st.title("🖌️ Digit Classification using SVM")
# st.write("Draw a digit (0-9) below and let SVM predict it!")

# # Create a drawing canvas
# canvas = st.empty()
# image = st.file_uploader("Upload an image of a digit (28x28 grayscale)", type=["png", "jpg", "jpeg"])

# # If an image is uploaded
# if image:
#     file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (8, 8))  # Resize to match dataset format
#     img = 16 - (img // 16)  # Normalize to match sklearn digits dataset
    
#     # Show the uploaded image
#     st.image(img, caption="Resized Image", width=150)

#     # Flatten the image to a 64-pixel vector
#     img_flatten = img.flatten().reshape(1, -1)
    
#     # Scale features
#     img_scaled = scaler.transform(img_flatten)

#     # Predict digit
#     prediction = svm_model.predict(img_scaled)
#     st.write(f"🎯 Predicted Digit: **{prediction[0]}**")

#     # Show the confidence scores
#     confidence_scores = svm_model.decision_function(img_scaled)
#     st.bar_chart(confidence_scores[0])

# st.write("📌 Upload an image of a handwritten digit to classify it!")





import streamlit as st
import numpy as np
import joblib
import cv2
import matplotlib.pyplot as plt

# Load the SVM model and scaler
svm_model = joblib.load(r"E:/SVM_Model/SVM/Best_Model.pkl")  
scaler = joblib.load(r"E:/SVM_Model/SVM/scaler.pkl")  

# Streamlit UI
st.title("🖌️ Digit Classification using SVM")
st.write("Draw or upload a digit (0-9) below, and let the model predict it!")

# Upload image
image = st.file_uploader("📌 Upload a digit image (8x8 grayscale)", type=["png", "jpg", "jpeg"])

if image:
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (8, 8))  # Resize to match dataset format
    img = 16 - (img // 16)  # Normalize to match sklearn digits dataset

    # Show uploaded image
    st.image(img, caption="🖼 Resized Image", width=150)

    # Flatten and scale the image
    img_flatten = img.flatten().reshape(1, -1)
    img_scaled = scaler.transform(img_flatten)

    # Predict digit
    prediction = svm_model.predict(img_scaled)
    confidence_scores = svm_model.decision_function(img_scaled)

    # Display the result
    st.markdown(f"### 🎯 Predicted Digit: **{prediction[0]}**")

    # Show confidence scores as a sorted bar chart
    sorted_indices = np.argsort(confidence_scores[0])[::-1]  # Sort in descending order
    top_3_digits = sorted_indices[:3]  # Get top-3 predictions

    fig, ax = plt.subplots()
    ax.bar([str(d) for d in top_3_digits], confidence_scores[0][top_3_digits], color=['blue', 'green', 'red'])
    ax.set_xlabel("Digits")
    ax.set_ylabel("Confidence Score")
    ax.set_title("Top 3 Predictions")
    st.pyplot(fig)

st.write("📌 Upload an image of a handwritten digit to classify it!")
