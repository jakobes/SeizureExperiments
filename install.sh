#/usr/bin/env bash

cd ${FENICS_HOME}/fenicstools
${FENICS_PYTHON} -m pip install --user .

cd ${FENICS_HOME}/xalode
${FENICS_PYTHON} -m pip install --user .
