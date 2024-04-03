import streamlit as st
# Streamlit app
def main():
    st.title('COVID-19 Detection from Chest X-rays')
    st.write('Upload a chest X-ray image and click on "Predict"')

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        prediction = predict(uploaded_file)

        if prediction[0][0] > 0.5:
            st.write('Prediction: COVID-19 Positive')
        else:
            st.write('Prediction: COVID-19 Negative')

if __name__ == '__main__':
    main()
