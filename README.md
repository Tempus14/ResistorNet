# ResistorNet
Collection of scripts used for the Resistor Net analogy
## Matrix-Inversionstest
This script is just a little helper in visualisation of inverting Resistor- and Conductance-Matrices. 
This was used while trying to understand how such a matrix has to be constructed. 
This docment serves no practical use for the final project.
## Network-Solver
This script contians a working algorithm to calculate currents flowing trough resistors in a given network.
The network geometry, the nodal voltages and the resistor values have to be known.
## NetworkBuild
This document summarizes the process of building the network
## Network-Builder
This document was used to test the performance of the algorithm in NetworkBuild and find the suprising result, that one combined for-loop is faster than individual networkx-functions.
## Combined
This file is a short collection of the relevant part of the code from Network-Solver and Network-Builder.
