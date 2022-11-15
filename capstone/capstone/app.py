import logging
import time
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import streamlit as st

# TODO
# - [-] connect to aws ?
# - [ ] call api? and get patient specific
# - [ ] test with multiple patients?
# - [ ] make a prediction csv
# - [-] deploy and share
# - [ ] pick good and bad patient and model results
# - [ ] add bounding boxes to animation
# - [ ] create two screen capture - good and bad
# - [x] add more patients to dropdown
# - [x] add view for axial animation
# - [x] use a dicom viewer https://neurosnippets.com/posts/diesitcom/


@st.cache
def get_frac_prob(patient):
    """
    Return probability of fracture [patient overall, C1,C2,C3,C4,C5,C6,C7]
    """

    np.random.seed(int(patient))

    # DEBUG patient
    vert_prob = np.random.uniform(low=0.0, high=100.0, size=7)
    if any(vert_prob > 50):
        prob = np.random.uniform(low=50.0, high=100.0, size=1)
    else:
        prob = np.random.uniform(low=0.0, high=50.0, size=1)
    prob = np.concatenate((prob, vert_prob))

    prob_df = pd.Series(
        prob,
        index=["Patient Overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
        name="Probability",
    )
    prob_df = prob_df.map(lambda x: "{:.1f}%".format(x)).to_frame()
    return prob_df


def get_sagittal_view(patient):
    """
    Get sagittal view for patient
    """
    fname = Path(f"img/sagittal_train_image/1.2.826.0.1.3680043.{patient}.png")
    fname = p / fname
    image = mpimg.imread(fname)
    st.image(image, use_column_width=True)


def init_dicom_reader(dir):
    """ "
    Instantiate DICOM reader
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dir)
    reader.SetFileNames(dicom_names)
    reader.LoadPrivateTagsOn()
    reader.MetaDataDictionaryArrayUpdateOn()
    data = reader.Execute()
    img = sitk.GetArrayViewFromImage(data)
    n_slices = img.shape[0]
    return img, n_slices


def plot_slice(vol, slice_ix):
    """
    Define a CT slice and plot
    """
    fig, ax = plt.subplots()
    plt.axis("off")
    selected_slice = vol[slice_ix, :, :]
    ax.imshow(selected_slice, origin="lower", cmap="bone")
    return fig


def get_axial_view(patient):
    """
    Animate through axial slices.
    """

    fname = Path(f"img/train_image/1.2.826.0.1.3680043.{patient}")
    img, n_slices = init_dicom_reader(dir=str(p / fname))

    for frame_num, frame_index in enumerate(np.linspace(0, n_slices, 100)):

        frame_text.text(f"Frame {int(frame_index)+1}/{n_slices}")
        progress_bar.progress(frame_num)
        if frame_index < n_slices:
            fig = plot_slice(img, int(frame_index))
            image.pyplot(fig)

            time.sleep(0.5)

    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    p = Path("/Users/janehung/berkeley-mids/w210/capstone/capstone/")

    st.set_page_config(
        layout="wide",
        page_title="NeckFrac",
        page_icon=str(p / Path("img/neckfrac_logo.jpeg")),
    )
    st.title("🦴 NeckFrac - Quicker, better, more accurate diagnosis to save lives.")

    col1, col2 = st.columns(2)

    with st.sidebar:
        patient = st.selectbox(
            "Patient ID", ("13096", "13097", "13098", "13099", "13100")
        )
        logging.info(f"Getting data for Patient ID: {patient}")

        view = st.radio("Anatomical Plane", ("Sagittal", "Axial"), horizontal=True)
        logging.info(f"Rendering {view} view")

        if view == "Axial":
            progress_bar = st.progress(0)
            frame_text = st.empty()

    with col1:
        st.subheader(f"{view} View")
        if view == "Sagittal":
            try:
                get_sagittal_view(patient)
            except:
                st.write("Come back later 😅")
        elif view == "Axial":
            image = st.empty()
            try:
                get_axial_view(patient)
            except:
                st.write("Come back later 😅")

            # progress_bar.empty()
            # frame_text.empty()

    with col2:
        st.subheader("Probability of Cervical Fracture")
        try:
            prob_df = get_frac_prob(patient)
            st.dataframe(prob_df, use_container_width=True)
        except:
            st.write("Come back later 😅")
