Lines 267, 268, 269 of LS_Assembly.hpp has:

evaluatePrecomputations(), writeMToFile(), getNeighborInteractions()

These functions do some precomputations and store the results in files. A folder precomputations is created where the necessary things are stored. So these lines can be commented out when you rerun the code to save time. When you give a different set of inputs to the LS problem, then these lines need to be uncommented, as new data needs to be written.

Make sure Eigen, Boost, openmp libraries are there in the system and linked.

It takes these inputs at run-time:

nChebNodes : number of gridPoints in leaf box in 1D

treeAdaptivity : tolerance for tree adpativity

kappa : frequency of operation

degreeOfBases : for discretization of Lippmann-Schwinger equation

L	: domain size

Qchoice	: contrast function choice


To run the code input in terminal:

make -f Makefile_LS.mk clean

make -f Makefile_LS.mk

./testKernel 6 5 40.0 6 0.5 0
