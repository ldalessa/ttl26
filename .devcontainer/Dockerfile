# syntax=docker/dockerfile:1-labs
FROM debian:testing

RUN apt-get update
RUN apt-get -y full-upgrade
RUN apt-get -y install wget curl sudo make
#    lsb-release software-properties-common gnupg procps make valgrind sudo

RUN useradd -rms /bin/bash -G sudo -p "$(openssl passwd -6 devcontainer)" devcontainer
RUN passwd -d devcontainer

# install llvm
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 18 all && \
    rm ./llvm.sh

RUN arch="$(uname -m)"; version="$(curl -s https://api.github.com/repos/Kitware/CMake/releases/latest | jq -r '.tag_name' | cut -c 2-)"; echo "Latest Version: $version" && \
    url="https://github.com/Kitware/CMake/releases/download/v$version/cmake-$version-linux-$arch.sh"; echo "Download URL: $url" && \
    wget -q $url && \
    sh cmake-$version-linux-$arch.sh --prefix=/usr/local --skip-license && \
    rm -rf cmake-$version-linux-$arch.sh

USER devcontainer

ENV PATH="/usr/lib/llvm-18/bin:$PATH" \
    CC=clang \
    CXX=clang++
