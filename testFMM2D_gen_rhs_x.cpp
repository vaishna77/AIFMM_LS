//
//  testFMM2D.cpp
//
//
//  Created by Vaishnavi Gujjula on 1/4/21.
//
//
#include "kernel.hpp"
#include "ACA.hpp"
#include "FMM2DTree_gen_rhs_x.hpp"
// #include "KDTree.cpp"

int main(int argc, char* argv[]) {
	// unsigned N                  = atoi(argv[1]);  // Number of particles.

	int nChebNodes			=	atoi(argv[1]);//number of gridPoints in leaf box in 1D
	int treeAdaptivity	=	1;//tolerance for tree adpativity
	kappa 							= atof(argv[2]);//frequency of operation
	int degreeOfBases 	= atoi(argv[3]); //for discretization of Lippmann-Schwinger equation
	double L						=	atof(argv[4]); //domain size
	Qchoice		    			=	atoi(argv[5]); // contrast function choice
	unsigned MinParticlesInLeaf = 4; // minimum particles in each leaf of KD Tree
	int TOL_POW                 = atoi(argv[6]); // tolerance for compressions in powers of 10
	int nLevelsUniform = atoi(argv[7]);

	std::cout << std::endl << "---------------problem parameters---------------" << std::endl;
	std::cout << "kappa: " << kappa << std::endl;
  std::cout << "nChebNodes: " << nChebNodes << std::endl;
	std::cout << "L: " << L << std::endl;
  std::cout << "Qchoice: " << Qchoice << std::endl;
  std::cout << "TOL_POW: " << TOL_POW << std::endl;
  std::cout << "nLevelsUniform: " << nLevelsUniform << std::endl;
  std::cout << "------------------------------------------------" << std::endl << std::endl;

	std::vector<pts2D> particles_X, particles_Y; //dummy variables
	userkernel* mykernel		=	new userkernel(nChebNodes, treeAdaptivity, degreeOfBases, L, particles_X, particles_Y, nLevelsUniform);
	unsigned N = mykernel->N;
	// unsigned nLevels            = log(N/MinParticlesInLeaf)/log(4);// for KD tree
	unsigned nLevels            = mykernel->nLevelsResolve;//for quad tree

	double start, end;
	unsigned n_Dimension    =       2;  //      Dimension.
	unsigned n_Properties   =       1;  //      Number of properties/rhs.
	double* locations       =       new double [N*n_Dimension];   //      Stores all the locations.
	dtype* properties      =       new dtype [N*n_Properties];  //      Stores all the properties.

	// Generates random locations and random values of property. Change the locations and rhs as per your need
	/////////////////////////////// GIVE INPUTS HERE ///////////////////////////////////////////////
	unsigned count_Location =       0;
	unsigned count_Property =       0;
	for (unsigned j=0; j<N; ++j) {
			for (unsigned k=0; k<n_Dimension; ++k) {
				if(k == 0) {
					locations[count_Location]       =       mykernel->gridPoints[j].x;//2*(int(rand())%2)-1;//double(rand())/double(RAND_MAX);
					++count_Location;
				}
				else {
					locations[count_Location]       =       mykernel->gridPoints[j].y;//2*(int(rand())%2)-1;//double(rand())/double(RAND_MAX);
					++count_Location;
				}
			}
			for (unsigned k=0; k<n_Properties; ++k) {
					// properties[count_Property]      =       10.0*double(rand())/double(RAND_MAX)-5.0;// + I*(10.0*double(rand())/double(RAND_MAX)-5.0);
					properties[count_Property]      =       mykernel->rhs(j);//10.0*double(rand())/double(RAND_MAX)-5.0 + I*(10.0*double(rand())/double(RAND_MAX)-5.0);
					++count_Property;
			}
	}
  //////////////////////////////// KD Tree //////////////////////////////////////////
	// std::vector<std::vector<int> > boxNumbers; // boxNumbers[nLevels] contains box numbers in N ordering. ex: [0 3 1 2]
  // double* sorted_Locations        =       new double[N*n_Dimension];
  // dtype* sorted_Properties       =       new dtype[N*n_Properties];
  // std::vector<int> NumberOfParticlesInLeaves;// contains number of particels in each leaf in N ordering
	// Creates a KDTree given the locations. This KD Tree class generates a uniform tree - all leaves are at level nLevels. Number of particles in boxes at a given level differ by atmost 1.
	// sort_KDTree(N, n_Dimension, locations, n_Properties, properties, MinParticlesInLeaf, nLevels, sorted_Locations, sorted_Properties, boxNumbers, NumberOfParticlesInLeaves);

	// quad tree
	std::vector<int> boxNumbers; // boxNumbers[nLevels] contains box numbers in N ordering. ex: [0 3 1 2]
  std::vector<int> NumberOfParticlesInLeaves;// contains number of particels in each leaf in N ordering
	for (size_t i = 0; i < pow(4,nLevels); i++) {
		boxNumbers.push_back(i);
		NumberOfParticlesInLeaves.push_back(nChebNodes*nChebNodes);
	}
	std::cout << "N: " << N << std::endl;
  //////////////////////////////// FMM2D //////////////////////////////////////////
	FMM2DTree<userkernel>* A	=	new FMM2DTree<userkernel>(mykernel, int(N), int(nLevels), TOL_POW, locations, properties, boxNumbers, NumberOfParticlesInLeaves);
	// FMM2DTree<userkernel>* A	=	new FMM2DTree<userkernel>(mykernel, int(N), int(nLevels), TOL_POW, sorted_Locations, sorted_Properties, boxNumbers[nLevels], NumberOfParticlesInLeaves);
	//////////////////////// CREATE TREE AND INITIALISE /////////////////////////////
	start	=	omp_get_wtime();
	A->createTree();
	A->assign_Tree_Interactions();
	end		=	omp_get_wtime();
	double timeCreateTree	=	(end-start);
  std::cout << "Number of Levels of tree: " << nLevels << std::endl;
	std::cout << std::endl << "Time taken to create the tree is: " << timeCreateTree << std::endl;

	//////////////////////// ASSEMBLE ////////////////////////////////////////
	start	=	omp_get_wtime();
	A->assignLeafChargeLocations();
	A->assign_Leaf_rhs();
	A->getNodes();
	// A->get_L2P_P2M();
	// A->check1();
	// std::cout << "here" << std::endl;
	A->assemble_M2L();
	// std::cout << "here" << std::endl;
	A->initialise_phase();
	// std::cout << "here" << std::endl;
	A->initialise_P2P_Leaf_Level();
	end		=	omp_get_wtime();
	double timeAssemble =	(end-start);
	std::cout << std::endl << "Time taken to assemble is: " << timeAssemble << std::endl;

	//////////////////////// ELIMINATE PHASE ////////////////////////////////////////
	start	=	omp_get_wtime();
	// A->eliminate_phase();
	A->eliminate_phase_efficient();
	end		=	omp_get_wtime();
	double timeFactorize =	(end-start);
	std::cout << std::endl << "Time taken to factorize (eliminate phase): " << timeFactorize << std::endl;

	//////////////////////// BACK SUBSTITUTION PHASE ////////////////////////////////////////
	start	=	omp_get_wtime();
	A->back_substitution_phase();
	end		=	omp_get_wtime();
	double timeSolve =	(end-start);
	std::cout << std::endl << "Time taken to solve (back substitution phase): " << timeSolve << std::endl;

	//////////////////////// ERROR CALCULATION ////////////////////////////////////////
	// A->check();
	std::cout << std::endl << "Performing error calculation... " << std::endl;
	A->assign_Leaf_rhs();
	double error = A->error_check();
	std::cout << std::endl << "error: " << error << std::endl << std::endl;

	// delete sorted_Locations;
  // delete sorted_Properties;
	delete A;
}
