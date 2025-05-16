# syntax=docker/dockerfile:1-labs
FROM debian:experimental

ARG USER_NAME
ARG LLVM_VERSION=20

RUN apt -y update
RUN apt -y full-upgrade
RUN apt -y install wget sudo make ninja-build valgrind git cmake locales locales-all
RUN	apt -y install gawk flex bison libgmp-dev libmpfr-dev libmpc-dev python3-dev binutils perl libisl-dev libzstd-dev tar gzip bzip2
RUN apt -y install clang-$LLVM_VERSION clang-tidy-$LLVM_VERSION clang-format-$LLVM_VERSION clang-tools-$LLVM_VERSION llvm-$LLVM_VERSION-dev lld-$LLVM_VERSION lldb-$LLVM_VERSION llvm-$LLVM_VERSION-tools libomp-$LLVM_VERSION-dev libc++-$LLVM_VERSION-dev libc++abi-$LLVM_VERSION-dev libclang-common-$LLVM_VERSION-dev libclang-$LLVM_VERSION-dev libclang-cpp$LLVM_VERSION-dev libunwind-$LLVM_VERSION-dev libclang-rt-$LLVM_VERSION-dev libpolly-$LLVM_VERSION-dev

RUN useradd -rms /bin/bash -G sudo -p "$(openssl passwd -6 devcontainer)" $USER_NAME
RUN passwd -d ldalessa

USER $USER_NAME

ENV PATH="/usr/lib/llvm-20/bin:$PATH" \
	CONFIG_SHELL=/bin/bash \
    TERM=xterm-256color \
    EDITOR=emacs \
    GIT_EDITOR=emacs \
    LANG='en_US.UTF-8' \
    LANGUAGE='en_US:en' \
    LC_ALL='en_US.UTF-8'
