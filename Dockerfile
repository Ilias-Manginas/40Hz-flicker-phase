FROM jupyter/base-notebook

# Install scientific Python stack + MNE
RUN pip install \
    numpy \
    scipy \
    matplotlib \
    pandas \
    scikit-learn \
    jupyterlab \
    mne \
    nibabel \
    nilearn \
    pycircstat2 \
    ffmpeg \
    pycircstat2 \
    h5io \
    dipy
