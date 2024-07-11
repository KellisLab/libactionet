cd build_test

if [[ "$1" == "arm" ]]; then
    echo "INFO: Building for arm64"
    arch -arm64 cmake --build . --parallel 6
elif [[ "$1" == "x86" ]]; then
    echo "INFO: Building for x86_64"
    arch -x86_64 /usr/local/bin/cmake --build . --parallel 6
else
    echo "INFO: arch not specified"
    echo "INFO: Building for arm64"
    arch -arm64 cmake --build . --parallel 6
fi
