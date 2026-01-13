FROM quay.io/jupyter/base-notebook

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    scikit-learn \
    torch \
    torchvision \
    torchaudio \
    tensorflow \
    pandas

WORKDIR /home/jovyan/work
