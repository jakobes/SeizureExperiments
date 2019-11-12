FROM quay.io/fenicsproject/stable:latest

USER root

ENV FENICS_PYTHON=python3

RUN ${FENICS_PYTHON} -m pip install --upgrade pip && \
    ${FENICS_PYTHON} -m pip install tqdm h5py cppimport && \
    ${FENICS_PYTHON} -m pip install git+git://github.com/jakobes/xalbrain@js-2018 && \
    ${FENICS_PYTHON} -m pip install git+git://github.com/jakobes/xalpost && \
    ${FENICS_PYTHON} -m pip install git+git://github.com/mikaem/fenicstools

RUN cd ${FENICS_HOME} && \
    ${FENICS_PYTHON} -m pip uninstall -y fenicstools && \
    git clone https://github.com/jakobes/fenicstools.git
# RUN cd fenicstools && ${FENICS_PYTHON} -m pip install --user .
# RUN ${FENICS_PYTHON} -c "import fenicstools; fenicstools.Probe"

RUN cd ${FENICS_HOME} && \
    git clone --recursive https://github.com/jakobes/xalode.git
# RUN cd xalode && ${FENICS_PYTHON} -m pip install --user .
# RUN ${FENICS_PYTHON} -c "import xalode"

RUN git clone https://github.com/jakobes/SeizureExperiments.git && \
    cd SeizureExperiments && chmod +x install.sh
