import streamlit as st
import cv2
import keras

loaded_model = keras.models.load_model("mnist_model.keras")
def process_image(image):
    img = cv2.imread(image.name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    return img

# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a page:', 
                           ['Prediction', 'Code', 'About'])

if options == 'Prediction': # Prediction page
    st.title('Handwritten Digit Classification with Neural Networks')


    # User inputs: image
    image = st.file_uploader('Upload an image:', type=['jpg', 'jpeg', 'png'])
    if image is not None:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Predict'):
        img_array = process_image(image)
        prediction = loaded_model.predict(img_array).argmax()
        st.write(f'The predicted digit is: {prediction}')
    
            
elif options == 'Code':
    st.header('Code')
    # Add a button to download the Jupyter notebook (.ipynb) file
    notebook_path = 'handwritten_image_classifier.ipynb'
    with open(notebook_path, "rb") as file:
        btn = st.download_button(
            label="Download Jupyter Notebook",
            data=file,
            file_name="handwritten_image_classifier.ipynb",
            mime="application/x-ipynb+json"
        )
    st.write('You can download the Jupyter notebook to view the code and the model building process.')
    st.write('--'*50)

    st.header('GitHub Repository')
    st.write('You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Handwritten-Digit-Classification)')
    st.write('--'*50)
    
elif options == 'About':
    st.title('About')
    st.write('This web app is a simple handwritten digit classification model built using a neural network with the MNIST dataset.')
    st.write('The MNIST dataset is a collection of 70,000 small square 28x28 pixel grayscale images of handwritten single digits between 0 and 9.')
    st.write('The model is built using TensorFlow and Keras, and the web app is built using Streamlit.')
    
    st.write('--'*50)
    st.write('The web app is open-source. You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Handwritten-Digit-Classification)')
    st.write('--'*50)

    st.header('Contact')
    st.write('You can contact me for any queries or feedback:')
    st.write('Email: gokulnpc@gmail.com')
    st.write('LinkedIn: [Gokuleshwaran Narayanan](https://www.linkedin.com/in/gokulnpc/)')
    st.write('--'*50)
