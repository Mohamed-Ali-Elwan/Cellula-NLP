"""
app.py
-------
Streamlit UI for the toxicity classification service.
Run with: streamlit run app.py

Written in a plain, linear style: everything happens top to bottom in
main(), the same way a normal script runs. No lazy-loading tricks.
"""

import streamlit as st

import config
from classifiers.distilbert_classifier import DistilBERTClassifier
from imagecaption import ImageCaption
from processing.preprocessing import Preprocessing
from data.database_manager import DatabaseManager
from service import ToxicityService


# -----------------------------------------------------------------------
# This function builds everything the app needs: the text preprocessor,
# the trained classifier, the image captioner, and the CSV database.
# @st.cache_resource means Streamlit runs this function ONCE and reuses
# the same objects on every rerun, instead of reloading the models on
# every click (which would be very slow).
# -----------------------------------------------------------------------
@st.cache_resource
def load_service():
    preprocessing = Preprocessing(config.RAW_DATA_PATH)
    preprocessing.load_data()
    preprocessing.encode_labels()

    classifier = DistilBERTClassifier(num_labels=config.NUM_LABELS)
    classifier.load_saved_model(config.DISTILBERT_MODEL_DIR)

    captioner = ImageCaption()

    db = DatabaseManager(config.DATABASE_PATH)

    return ToxicityService(
        preprocessing=preprocessing,
        classifier=classifier,
        captioner=captioner,
        db=db,
    )


def main():
    # Page title shown in the browser tab, and page width setting.
    # Must be the first Streamlit command in the script.
    st.set_page_config(page_title="Toxic Content Classifier", layout="centered")

    # Big heading at the top of the page.
    st.title("NLP & LLM Dashboard")
    # Smaller heading underneath it.
    st.subheader("Toxic Content Classification")

    # Show a spinner while the models load (only happens once, thanks
    # to @st.cache_resource above).
    with st.spinner("Loading models..."):
        service = load_service()

    # Three tabs across the top of the page. Anything indented under
    # "with tab_x:" only shows up when that tab is selected.
    tab_text, tab_image, tab_db = st.tabs(["Text input", "Image input", "View database"])

    # ---------------------------------------------------------------
    # TAB 1: classify raw text typed in by the user.
    # ---------------------------------------------------------------
    with tab_text:
        # A multi-line text box. `text_input` holds whatever is
        # currently typed in it.
        text_input = st.text_area("Enter text to classify")

        # A button. st.button() returns True only on the one script
        # run that happens right after it was clicked.
        if st.button("Classify text", key="classify_text_btn"):
            if text_input.strip():
                label = service.classify_text(text_input)
                # Green success box.
                st.success(f"Classification: **{label}**")
            else:
                # Yellow warning box.
                st.warning("Please enter some text first.")

    # ---------------------------------------------------------------
    # TAB 2: upload an image, caption it, classify the caption.
    # ---------------------------------------------------------------
    with tab_image:
        # File upload widget. Returns None until a file is chosen.
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            # Show the uploaded image on the page.
            st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

            if st.button("Caption + classify image", key="classify_image_btn"):
                caption, label = service.classify_image(uploaded_file)
                st.write(f"Generated caption: *{caption}*")
                st.success(f"Classification: **{label}**")

    # ---------------------------------------------------------------
    # TAB 3: show everything stored in the CSV database so far.
    # ---------------------------------------------------------------
    with tab_db:
        st.subheader("Stored records")

        history = service.get_history()

        # Displays a pandas DataFrame as an interactive table.
        st.dataframe(history, use_container_width=True)

        # Small gray footnote text.
        st.caption(f"Total records: {len(history)}")


# Standard Python entry point: only run main() if this file is executed
# directly (which is what `streamlit run app.py` does).
if __name__ == "__main__":
    main()