#!/bin/zsh
nvcc $1_*.cu -o main.run -lglut -lGL
./main.run
rm main.run
