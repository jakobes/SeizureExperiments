Bootstrap: docker
From: quay.io/fenicsproject/stable:2019.1.0.r3

%post
    sudo apt-get install libmpich-dev libhdf5-mpich-dev mpich
    ldconfig        # copied from fenics docker
    python3 -m pip install --upgrade pip
    python3 -m pip install tqdm h5py cppimport && \

    python3 -m pip install git+git://github.com/jakobes/xalbrain@js-2018 && \
    python3 -m pip install git+git://github.com/jakobes/xalpost && \
    python3 -m pip install git+git://github.com/jakobes/xalode

%runscript
    exec /bin/bash -i
