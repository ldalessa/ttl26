# syntax=docker/dockerfile:1-labs
FROM debian:testing

ARG USER_NAME

RUN apt-get update
RUN apt-get -y full-upgrade
RUN apt-get -y install wget sudo make ninja-build valgrind git
RUN apt-get -y install lsb-release software-properties-common gnupg # llvm.sh

# install llvm
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 18 all && \
    rm ./llvm.sh

# download and install cmake
RUN arch="$(uname -m)"; \
    version="3.30.2"; \
    url="https://github.com/Kitware/CMake/releases/download/v$version/cmake-$version-linux-$arch.sh"; \
    wget -q $url && \
    sh cmake-$version-linux-$arch.sh --prefix=/usr/local --skip-license && \
    rm cmake-$version-linux-$arch.sh


RUN useradd -rms /bin/bash -G sudo -p "$(openssl passwd -6 devcontainer)" $USER_NAME
RUN passwd -d ldalessa

USER $USER_NAME

ENV PATH="/usr/lib/llvm-18/bin:$PATH" \
    CC=clang \
    CXX=clang++ \
    TERM=xterm-256color \
    EDITOR=emacs \
    GIT_EDITOR=emacs \
    LANG='en_US.UTF-8' \
    LANGUAGE='en_US:en' \
    LC_ALL='en_US.UTF-8'
