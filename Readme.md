This is a 2D Algebraic Inverse FMM Code (AIFMM) - a direct solver for matrices arising out of PDEs. Solves for x in Ax=b.

It is a completely algebraic version. Compressions are made via ACA.

It is applicable to all FMM-able kernels.

Follows the KD tree data structure.

The kernel that defines the matrix A is to be defined in "kernel.hpp" file. RHS b and the location of charges are to be defined in the "testFMM2D.cpp" file. The respective variables are "locations" and "properties".

"ACA.hpp" file contains the ACA module.

"FMM2DTree.hpp" file contains the algorithm.

"testFMM2D.cpp" is the top level file.

"KDTree.hpp" and "KDTree.cpp" are the files where KD Tree routines are defined

Before running make sure CPP compiler, Eigen and openmp paths are specified in Makefile.

It takes these inputs at run time: Number of Particles(size of system), minimum particles in a leaf of KD Tree, tolerance for compressions in powers of 10

To run it input in terminal:

make -f Makefile2D.mk clean

make -f Makefile2D.mk

./testFMM2D 16384 16 10

The output looks like...

Number of Levels of tree: 5

Time taken to create the tree is: 0.003684

Time taken to assemble is: 5.18255

Time taken to factorize (eliminate phase): 185.7

Time taken to solve (back substitution phase): 1.78744

Performing error calculation...

error: 5.01998e-05
