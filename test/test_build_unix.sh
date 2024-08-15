#!/usr/bin/env bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "${SCRIPT_DIR}"/.. || exit

cd build_test || exit

cmake --build . --parallel 6
