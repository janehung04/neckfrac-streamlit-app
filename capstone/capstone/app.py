import logging
import time
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# TODO
# - [ ] connect to aws
# - [ ] make a prediction csv
# - [x] add view for axial animation
# - [ ] use a dicom viewer https://neurosnippets.com/posts/diesitcom/


@st.cache
def get_frac_prob():
    """
    Return probability of fracture [patient overall, C1,C2,C3,C4,C5,C6,C7]
    """
    # TODO call api? and get patient specific
    prob = np.zeros(8)
    prob[0] = 100
    prob[[1, 2]] = 100
    prob_df = pd.Series(
        prob,
        index=["Patient Overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
        name="Probability",
    )
    prob_df = prob_df.map(lambda x: "{:.1f}%".format(x)).to_frame()
    return prob_df


def get_sagittal_view(patient):
    fname = Path(f"img/sagittal_train_image/1.2.826.0.1.3680043.{patient}.png")
    fname = p / fname
    image = mpimg.imread(fname)
    st.image(image, use_column_width=True)


def get_dcm_images(path):
    paths = list(path.glob("*"))
    paths.sort(
        key=lambda x: int(x.stem)
    )  # sort based on slice index which is the filename: index.dcm
    data = [pydicom.dcmread(f) for f in paths]
    images = [apply_voi_lut(dcm.pixel_array, dcm) for dcm in data]
    return images


def get_axial_view(patient):
    fname = Path(f"img/train_image/1.2.826.0.1.3680043.{patient}")
    dcm_images = get_dcm_images(path=p / fname)
    # range(len(dcm_images))
    for frame_num, frame_index in enumerate(np.linspace(0, len(dcm_images), 100)):
        # frame_text.text("Frame %i/100" % (frame_num + 1))
        frame_text.text(f"Frame {int(frame_index)+1}/{len(dcm_images)}")
        progress_bar.progress(frame_num)

        image.image(
            dcm_images[int(frame_index)], use_column_width=True, clamp=True, channels="RGB"
        )
        time.sleep(0.5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    p = Path("/Users/janehung/berkeley-mids/w210/capstone/capstone/")

    st.set_page_config(
        layout="wide",
        page_title="NeckFrac",
        page_icon=str(p / Path("img/neckfrac_logo.jpeg")),
    )
    st.title("🦴 NeckFrac - Quicker, better, more accurate diagnosis to save lives.")

    col1, col2 = st.columns([2, 1])

    with st.sidebar:
        patient = st.selectbox("Patient ID", ("13096", "XXXX"))
        logging.info(f"Getting data for Patient ID: {patient}")

        view = st.radio("Anatomical Plane", ("Sagittal", "Axial"), horizontal=True)
        logging.info(f"Rendering {view} view")

        if view == "Axial":
            progress_bar = st.progress(0)
            frame_text = st.empty()

    with col1:
        st.subheader(f"{view} View")
        if view == "Sagittal":
            get_sagittal_view(patient)
        elif view == "Axial":
            image = st.empty()
            get_axial_view(patient)

            # progress_bar.empty()
            # frame_text.empty()

    with col2:
        st.subheader("Probability of Cervical Fracture")
        prob_df = get_frac_prob()
        st.dataframe(prob_df, use_container_width=True)
