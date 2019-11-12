#/usr/bin/env bash

cd ${FENICS_HOME}/fenicsproject
${FENICS_HOME} -m pip install --user .

cd ${FENICS_HOME}/xalode
${FENICS_HOME} -m pip install --user .
