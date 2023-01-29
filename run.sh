#!/bin/zsh
nvcc $1_*.cu -o main.run -lglut -lGL --Wno-deprecated-declarations
./main.run
rm main.run
