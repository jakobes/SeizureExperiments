#/usr/bin/env bash

git clone https://github.com/jakobes/fenicstools.git
cd fenicstools
${FENICS_PYTHON} -m pip install --user .
${FENICS_PYTHON} -c "import fenicstools; fenicstools.Probe"
