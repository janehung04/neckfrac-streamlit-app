import ast
import logging
import time
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import streamlit as st


def get_logo():
    """
    Get logo for when no patient is selected
    """
    fname = Path(f"NeckFrac_white_background.png")
    fname = p / fname
    image = mpimg.imread(fname)
    return image


@st.cache
def get_frac_prob(patient):
    """
    Return probability of fracture [patient overall, C1,C2,C3,C4,C5,C6,C7]
    """
    all_prob = {
        "13096": [1, 0, 0, 0, 0, 0, 1, 1],
        "9447": [1, 0, 0, 0, 1, 1, 1, 1],
        "25651": [1, 0, 1, 0, 1, 1, 1, 1],
        "13444": [1, 0, 1, 0, 0, 0, 1, 0],
    }
    prob = all_prob.get(str(patient), np.zeros(8).astype(int).tolist())
    prob_df = pd.Series(
        prob,
        index=["Patient Overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
        name="Fractured?",
    )
    prob_df = prob_df.map({1: "YES", 0: ""}).to_frame()
    return prob_df


def get_sagittal_view(patient):
    """
    Get sagittal view for patient
    """
    fname = Path(f"sagittal_train_image/1.2.826.0.1.3680043.{patient}.png")
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


def get_bounding_box(slice_bb):
    # Calc bounding box info
    anchor = (slice_bb.x.values[0], slice_bb.y.values[0])
    height = slice_bb.height.values[0]
    width = slice_bb.width.values[0]
    return anchor, height, width


def plot_slice(vol, slice_ix, slice_bb):
    """
    Define a CT slice and plot
    """
    fig, ax = plt.subplots()
    plt.axis("off")
    selected_slice = vol[slice_ix, :, :]
    ax.imshow(selected_slice, cmap="bone")

    if slice_bb.shape[0] != 0:
        # Add bounding box to the Axes
        anchor, height, width = get_bounding_box(slice_bb)
        rect = patches.Rectangle(
            anchor, width, height, linewidth=1, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)

    return fig


def get_axial_view(patient):
    """
    Animate through axial slices.
    """

    # Get DICOM image
    fname = Path(f"train_image/1.2.826.0.1.3680043.{patient}")
    img, n_slices = init_dicom_reader(dir=str(p / fname))

    # Get bounding box info
    bb_fname = p / Path("train_bounding_boxes.csv")
    bb_df = pd.read_csv(bb_fname)
    bb_df = bb_df[bb_df.StudyInstanceUID.str.endswith(str(patient))]

    for frame_num, frame_index in enumerate(np.linspace(0, n_slices, 100)):

        frame_text.text(f"Frame {int(frame_index)+1}/{n_slices}")
        progress_bar.progress(frame_num)
        if frame_index < n_slices:
            slice_bb = bb_df[bb_df.slice_number.astype(int) == int(frame_index)]
            fig = plot_slice(img, int(frame_index), slice_bb)
            image.pyplot(fig)

            time.sleep(0.5)
        plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    p = Path("/Users/janehung/berkeley-mids/w210/data")

    st.set_page_config(
        layout="wide",
        page_title="NeckFrac",
        page_icon=str(p / Path("NeckFrac_white_background.png")),
    )
    st.title("🦴 NeckFrac - Quicker, better, more accurate diagnosis to save lives.")

    with st.sidebar:
        patient = st.selectbox(
            "Patient ID",
            ("Select one", "13096", "9447", "25651", "13444", "13100"),
        )
        logging.info(f"Getting data for Patient ID: {patient}")

        view = st.radio("Anatomical Plane", ("Sagittal", "Axial"), horizontal=True)
        logging.info(f"Rendering {view} view")

        if view == "Axial":
            progress_bar = st.progress(0)
            frame_text = st.empty()

    if patient == "Select one":
        col1, col2, col3 = st.columns(3)
        col2.image(get_logo(), use_column_width=True)

    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{view} View")
            if view == "Sagittal":
                try:
                    get_sagittal_view(patient)
                except Exception as e:
                    st.write("Patient not found 😅")
            elif view == "Axial":
                image = st.empty()
                try:
                    get_axial_view(patient)
                except Exception as e:
                    st.write("Patient not found 😅")

        with col2:
            st.subheader("Predicted Cervical Fracture")
            try:
                prob_df = get_frac_prob(patient)
                st.dataframe(prob_df, use_container_width=True)
            except Exception as e:
                st.write("Patient not found 😅")
