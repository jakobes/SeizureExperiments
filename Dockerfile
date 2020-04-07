# FROM quay.io/fenicsproject/stable
FROM quay.io/fenicsproject/stable:2019.1.0.r3

USER root

ENV FENICS_PYTHON=python3

RUN ${FENICS_PYTHON} -m pip install --upgrade pip && \
    ${FENICS_PYTHON} -m pip install tqdm h5py cppimport && \
    ${FENICS_PYTHON} -m pip install git+git://github.com/jakobes/xalbrain@d0b51158afd73801a4f7ae13dc8a2c8d36cd8192 && \
    ${FENICS_PYTHON} -m pip install git+git://github.com/jakobes/xalpost@d0b51158afd73801a4f7ae13dc8a2c8d36cd8192 && \
    ${FENICS_PYTHON} -m pip install git+git://github.com/jakobes/xalode@d0b51158afd73801a4f7ae13dc8a2c8d36cd8192

RUN cd ${FENICS_HOME} && \
    git clone https://github.com/jakobes/fenicstools.git

RUN git clone https://github.com/jakobes/SeizureExperiments.git

ADD install.sh
