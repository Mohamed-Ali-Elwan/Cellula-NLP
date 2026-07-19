import streamlit as st

from service import ClassificationService


service = ClassificationService(r"data\cellula toxic data  (1).csv")

st.title("Cellula Toxic Content Classification")

option = st.sidebar.selectbox(

    "Choose Input",

    [

        "Text",

        "Image",

        "Database"

    ]

)

##########################################################
# TEXT
##########################################################

if option == "Text":

    st.header("Text Classification")

    text = st.text_area(
        "Enter your text"
    )

    if st.button("Predict"):

        prediction = service.classify_text(
            text
        )

        st.success(
            f"Prediction: {prediction}"
        )

##########################################################
# IMAGE
##########################################################

elif option == "Image":

    st.header("Image Classification")

    uploaded_image = st.file_uploader(

        "Upload Image",

        type=["jpg", "jpeg", "png"]

    )

    if uploaded_image is not None:

        with open("temp_image.jpg", "wb") as f:

            f.write(uploaded_image.getbuffer())

        st.image(uploaded_image)

        if st.button("Generate Caption & Predict"):

            caption, prediction = service.classify_image(
                "temp_image.jpg"
            )

            st.write("Caption:")

            st.info(caption)

            st.success(
                f"Prediction: {prediction}"
            )

##########################################################
# DATABASE
##########################################################

else:

    st.header("Database")

    data = service.get_database()

    st.dataframe(data)