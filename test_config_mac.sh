cd build_test

if [[ "$1" = "arm" ]]; then
    echo "INFO: Configuring for arm64"
    arch -arm64 cmake ..
elif [[ "$1" = "x86" ]]; then
    echo "INFO: Configuring for x86_64"
    arch -x86_64 /usr/local/bin/cmake ..
else
    echo "INFO: arch not specified"
    echo "INFO: Configuring for arm64"
    arch -arm64 cmake ..
fi
