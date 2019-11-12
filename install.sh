#/usr/bin/env bash

cd ${FENICS_HOME}/fenicstols
${FENICS_PYTHON} -m pip install --user .

cd ${FENICS_HOME}/xalode
${FENICS_PYTHON} -m pip install --user .
