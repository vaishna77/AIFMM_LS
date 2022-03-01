This library solves for Electromagnetic scattering in 2D.

It is a 2D Algebraic Inverse FMM Code (AIFMM) - a direct solver for matrices arising out of the Lippmann-Schwinger equation in 2D.

It is a completely algebraic version. Compressions are made via NCA.

The kernel that defines the matrix A is to be defined in "kernel.hpp" file. RHS b and the location of charges are to be defined in the "testFMM2D.cpp" file. The respective variables are "locations" and "properties".

"ACA.hpp" file contains the ACA module.

Before running make sure CPP compiler, Eigen, cereal and openmp paths are specified in Makefile.

It takes these inputs at run time: number of chebyshev nodes in a single Dimension, wavenumber, degree of polynomial basis functions, half width of domain, choice of contrast function, tolerance for compressions in negative powers of 10

To run it input in terminal:

make -f Makefile2D.mk clean

make -f Makefile2D.mk

./testFMM2D_find_error 6 10 6 1 0 8 5
