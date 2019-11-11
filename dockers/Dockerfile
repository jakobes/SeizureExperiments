FROM quay.io/fenicsproject/stable:latest

USER root

ENV FENICS_PYTHON=python3

RUN ${FENICS_PYTHON} -m pip install --upgrade pip && \
    ${FENICS_PYTHON} -m pip install tqdm && \
    ${FENICS_PYTHON} -m pip install git+git://github.com/jakobes/xalbrain && \
    ${FENICS_PYTHON} -m pip install git+git://github.com/jakobes/xalpost && \
    ${FENICS_PYTHON} -m pip install git+git://github.com/mikaem/fenicstools && \
    git clone --recursive https://github.com/jakobes/xalode.git && \
    cd xalode && ${FENICS_PYTHON} -m pip install . && \
    cd && \
    git clone https://github.com/jakobes/SeizureExperiments.git && \
    cd SeizureExperiments
