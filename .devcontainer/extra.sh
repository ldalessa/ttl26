#!/bin/bash

emacsd=$HOME/.emacs.d
if ! test -f $emacsd/init.el
then
    echo "Copying init.el"
    mkdir -p $emacsd  && cp init.el "$_"
else
    echo "Not copying init.el"
fi

sudo apt -y update
sudo apt -y full-upgrade
sudo apt -y autoremove
sudo apt -y install build-essential gcc g++ gdb man-db emacs elpa-cmake-mode ispell
