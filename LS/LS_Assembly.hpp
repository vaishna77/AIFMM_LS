#ifndef __LS_Assembly_hpp__
#define __LS_Assembly_hpp__

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <bits/stdc++.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>
#include "Eigen/Dense"
#include <boost/math/special_functions/bessel.hpp>
#include <filesystem>
#include <unistd.h>

#ifdef USE_COMPLEX64
    using dtype=std::complex<double>;
    using dtype_base=double;
    using Mat=Eigen::MatrixXcd;
    using Vec=Eigen::VectorXcd;
    const std::complex<double> I(0.0, 1.0);
#endif

const double epsilonRoundOff = pow(10,-16);
// #define EIGEN_DONT_PARALLELIZE
using namespace std;
using namespace Eigen;
const double PI	=	3.1415926535897932384;
int Qchoice;
string currentDirectory;
typedef MatrixXcd Mat;
typedef VectorXcd Vec;
double kappa;
struct pts2D {
	double x,y;
};
double besselJ(int n, double x) {
	if (n >= 0) {
		double temp = boost::math::cyl_bessel_j(double(n), x);
		return temp;
	}
	else {
		double temp = boost::math::cyl_bessel_j(double(-n), x);
		if (-n%2 == 0)
			return temp;
		else
			return -temp;
	}
}

double besselY(int n, double x) {
	if (n >= 0) {
		double temp = boost::math::cyl_neumann(double(n), x);
		return temp;
	}
	else {
		double temp = boost::math::cyl_neumann(double(-n), x);
		if (-n%2 == 0)
			return temp;
		else
			return -temp;
	}
}
#include "domain2D.hpp"
#include "FarFieldInteraction.hpp"
#include "FarFieldInteraction2.hpp"
#include "NearFieldInteraction1.hpp"
#include "NearFieldInteraction2.hpp"
#include "LS_functions.hpp"

pts2D SFN_centers[20] = {{-2.5, -2.5},//separatedFineNeighbors
												{-1.5, -2.5},
												{-0.5, -2.5},
												{0.5, -2.5},
												{1.5, -2.5},
												{2.5, -2.5},
												{2.5, -1.5},
												{2.5, -0.5},
												{2.5, 0.5},
												{2.5, 1.5},
												{2.5, 2.5},
												{1.5, 2.5},
												{0.5, 2.5},
												{-0.5, 2.5},
												{-1.5, 2.5},
												{-2.5, 2.5},
												{-2.5, 1.5},
												{-2.5, 0.5},
												{-2.5, -0.5},
												{-2.5, -1.5}
												};
pts2D N_centers[9] = {{-2, -2},//colleagueNeighbors
											 {0, -2},
											 {2, -2},
											 {2, 0},
											 {2, 2},
											 {0, 2},
											 {-2, 2},
											 {-2, 0},
											 {0, 0}};
pts2D FN_centers[12] = {{-1.5, -1.5},//fineNeighbors
												 {-0.5, -1.5},
												 {0.5, -1.5},
												 {1.5, -1.5},
												 {1.5, -0.5},
												 {1.5, 0.5},
												 {1.5, 1.5},
												 {0.5, 1.5},
												 {-0.5, 1.5},
												 {-1.5, 1.5},
												 {-1.5, 0.5},
												 {-1.5, -0.5}};
pts2D CN_centers[12] = {{-3, -3},//coarseNeighbors
												{-1, -3},
												{1, -3},
												{3, -3},
												{3, -1},
												{3, 1},
												{3, 3},
												{1, 3},
												{-1, 3},
												{-3, 3},
												{-3, 1},
												{-3, -1}};

class LS2DBox {
public:
	double radius;
	int active;
	bool ILActive;
	int level;
	int boxNumber;
	int parentNumber;
	int childrenNumbers[4];
	bool isLeaf;

	std::vector<orderedPair > neighborNumbers;

	int fineNeighbors[12];
	int coarseNeighbors[12];
	int separatedFineNeighbors[20];
	int colleagueNeighbors[9];

	LS2DBox () {
		boxNumber		=	-1;
		parentNumber	=	-1;
		for (int l=0; l<4; ++l) {
			childrenNumbers[l]	=	-1;
		}
		isLeaf = false;
		active = true;
		for (size_t i = 0; i < 12; i++) {
			fineNeighbors[i] = -1;
		}
		for (size_t i = 0; i < 12; i++) {
			coarseNeighbors[i] = -1;
		}
		for (size_t i = 0; i < 20; i++) {
			separatedFineNeighbors[i] = -1;
		}
		for (size_t i = 0; i < 9; i++) {
			colleagueNeighbors[i] = -1;
		}
	}

	pts2D center;
	std::vector<int> chargeLocations;
	std::vector<pts2D> chebNodes;
};

class LS2DTree : public userKernel {
public:
	double timeIn_getMatrixEntry;//for time profiling
	long NoIn_getMatrixEntry;
	int nLevels;			//	Number of levels in the tree.
	int nChebNodes;			//	Number of Chebyshev nodes along one direction.
	int rank;				//	Rank of interaction, i.e., rank = nChebNodes*nChebNodes.
	double L;				//	Semi-length of the simulation box.
	double smallestBoxSize;	//	This is L/2.0^(nLevels).
	double a;				//	Cut-off for self-interaction. This is less than the length of the smallest box size.
	const double A = 3.0; //higher A, higher level_FarField
	const double B = 0.35;//4.0; //smaller B, higher level_LFR
	int level_FarField;
	int nCones_LFR;
	double epsilonTree;//pow(10,-epsilonTree);
	std::vector<pts2D> particles_X;//locations
  std::vector<pts2D> particles_Y;//dummy values
	double SFN_angles[20];
	double N_angles[9];
	double FN_angles[12];
	double CN_angles[12];

	bool findPhi;
	string NeighborFilename;
	string MFilename;
	std::vector<pts2D> gridPoints;
	std::vector<int> nBoxesPerLevel;			//	Number of boxes at each level in the tree.
	std::vector<double> boxRadius;				//	Box radius at each level in the tree assuming the box at the root is [-1,1]^2
	std::vector<double> ConeAperture;
	std::vector<int> nCones;
	std::vector<double> boxHomogRadius;			//	Stores the value of boxRadius^{alpha}
	std::vector<double> boxLogHomogRadius;		//	Stores the value of alpha*log(boxRadius)
	std::vector<std::vector<LS2DBox> > tree;	//	The tree storing all the information.
	std::vector<std::vector<int> > indexTree;
	int treeAdaptivity;
	int yes2DFMM;
	std::vector<double> standardChebNodes1D;
	std::vector<pts2D> standardChebNodes;
	std::vector<pts2D> standardChebNodesChild;
	std::vector<pts2D> leafChebNodes;

	std::vector<pts2D> bottomTopBoundary;//used for plotting field
	std::vector<pts2D> leftRightBoundary;//used for plotting field

	std::vector<orderedPair> leafNodes;
	//	Different Operators
	Eigen::MatrixXd M2M[4];					//	Transfer from multipoles of 4 children to multipoles of parent.
	Eigen::MatrixXd L2L[4];					//	Transfer from locals of parent to locals of 4 children.
	FarFieldInteraction* FF; //object needed to computeFarField interaction of Lippmann Schwinger problem
	FarFieldInteraction2* FF2;
	std::vector<Mat > M; //M matrices computed during precomputations
	Eigen::MatrixXd Q_pinv;
	Eigen::MatrixXd Q_pinv2;
	int degreeOfBases;
	Mat colleagueNeighborInteraction[9][9];
	Mat fineNeighborInteraction[9][12];
	Mat separatedFineNeighborInteraction[9][20];
	Mat coarseNeighborInteraction[9][12];
	int nLevelsUniform;

// public:
	LS2DTree(int nCones_LFR, int nChebNodes, double L, int yes2DFMM, int TOL_POW, std::vector<pts2D> particles_X, std::vector<pts2D> particles_Y, double kappa, int degreeOfBases, int treeAdaptivity, int nLevelsUniform) : userKernel()
  {
		this->nLevelsUniform = nLevelsUniform;
		this->findPhi = true;
		this->NoIn_getMatrixEntry = 0;
		this->timeIn_getMatrixEntry = 0.0;
		this->degreeOfBases = degreeOfBases;//p minus 1
		this->treeAdaptivity = treeAdaptivity;
		this->epsilonTree = pow(10,-treeAdaptivity);
		this->FF = new FarFieldInteraction(nChebNodes, L, degreeOfBases);
		this->FF2 = new FarFieldInteraction2(nChebNodes, L, degreeOfBases);
		FF2->evaluateQ(Q_pinv2);
		this->nChebNodes		=	nChebNodes;
		this->rank			=	nChebNodes*nChebNodes;
		this->L				=	L;
		this->nCones_LFR		=	nCones_LFR;
		this->yes2DFMM = yes2DFMM;
		nBoxesPerLevel.push_back(1);
		boxRadius.push_back(L);
		this->a			=	smallestBoxSize;
		set_Standard_Cheb_Nodes();
  	get_Transfer_Matrix();
  	// createAdaptiveTree();
    // makeLevelRestriction_Tree();
    createUniformTree();
		outputAdaptiveGrid("tree.tex");
		assignLeafChargeLocations(gridPoints);//chargeLocations
    assignNonLeafChargeLocations();
    assignLeafChebNodes();
    FF->evaluateQ(Q_pinv);

		currentDirectory = std::filesystem::current_path();
		char final [512];
		sprintf (final, "%s/precomputations",currentDirectory.c_str());
		mkdir(final, 0775);
		//following 3 lines need to be uncommented when you change the inputs to LS equation
    evaluatePrecomputations(); //precomputations
    writeMToFile(); //precomputations
    getNeighborInteractions(); //precomputations

    readMFromFile();
    getNeighbors();
    readNeighborInteractions();
	}

	void evaluatePrecomputations() {
		FF->evaluateM(M);
	}

	void getIfNeighbor(int j, int k, int nj, int nk, int& typeOfNeighbor, int& NumOfNeighbor) {
		if ( fabs(tree[j][k].center.x - tree[nj][nk].center.x) >= 3.0*boxRadius[j] + boxRadius[nj]-epsilonRoundOff
		 || fabs(tree[j][k].center.y - tree[nj][nk].center.y) >= 3.0*boxRadius[j] + boxRadius[nj]-epsilonRoundOff) {
			 typeOfNeighbor = -1;
		}
		else {
			double angle = atan2(tree[nj][nk].center.y-tree[j][k].center.y, tree[nj][nk].center.x-tree[j][k].center.x);
			if (tree[j][k].radius == tree[nj][nk].radius) {
				typeOfNeighbor = 0;
				if (fabs(tree[nj][nk].center.y-tree[j][k].center.y)<epsilonRoundOff && fabs(tree[nj][nk].center.x-tree[j][k].center.x)<epsilonRoundOff) {
					NumOfNeighbor = 8;
				} // self
				else if (fabs(N_angles[0] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 0;
				}
				else if (fabs(N_angles[1] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 1;
				}
				else if (fabs(N_angles[2] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 2;
				}
				else if (fabs(N_angles[3] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 3;
				}
				else if (fabs(N_angles[4] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 4;
				}
				else if (fabs(N_angles[5] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 5;
				}
				else if (fabs(N_angles[6] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 6;
				}
				else if (fabs(N_angles[7] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 7;
				}
			}
			else if (tree[j][k].radius < tree[nj][nk].radius) { //coarse neighbors
				typeOfNeighbor = 1;
				if (fabs(CN_angles[0] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 0;
				}
				else if (fabs(CN_angles[1] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 1;
				}
				else if (fabs(CN_angles[2] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 2;
				}
				else if (fabs(CN_angles[3] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 3;
				}
				else if (fabs(CN_angles[4] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 4;
				}
				else if (fabs(CN_angles[5] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 5;
				}
				else if (fabs(CN_angles[6] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 6;
				}
				else if (fabs(CN_angles[7] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 7;
				}
				else if (fabs(CN_angles[8] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 8;
				}
				else if (fabs(CN_angles[9] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 9;
				}
				else if (fabs(CN_angles[10] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 10;
				}
				else if (fabs(CN_angles[11] - angle) < epsilonRoundOff) {
					NumOfNeighbor = 11;
				}
			}
			else { //fine neighbors and separated fine neighbors
				 if (find_distance(j, k, nj, nk) > 2.5*tree[j][k].radius) { //separated fine neighbors
					 typeOfNeighbor = 3;
					 if (fabs(SFN_angles[0] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 0;
					 }
					 else if (fabs(SFN_angles[1] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 1;
					 }
					 else if (fabs(SFN_angles[2] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 2;
					 }
					 else if (fabs(SFN_angles[3] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 3;
					 }
					 else if (fabs(SFN_angles[4] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 4;
					 }
					 else if (fabs(SFN_angles[5] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 5;
					 }
					 else if (fabs(SFN_angles[6] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 6;
					 }
					 else if (fabs(SFN_angles[7] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 7;
					 }
					 else if (fabs(SFN_angles[8] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 8;
					 }
					 else if (fabs(SFN_angles[9] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 9;
					 }
					 else if (fabs(SFN_angles[10] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 10;
					 }
					 else if (fabs(SFN_angles[11] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 11;
					 }
					 else if (fabs(SFN_angles[12] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 12;
					 }
					 else if (fabs(SFN_angles[13] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 13;
					 }
					 else if (fabs(SFN_angles[14] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 14;
					 }
					 else if (fabs(SFN_angles[15] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 15;
					 }
					 else if (fabs(SFN_angles[16] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 16;
					 }
					 else if (fabs(SFN_angles[17] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 17;
					 }
					 else if (fabs(SFN_angles[18] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 18;
					 }
					 else {
						 NumOfNeighbor = 19;
					 }
				 }
				 else {//fine neighbors
					 typeOfNeighbor = 2;
					 if (fabs(FN_angles[0] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 0;
					 }
					 else if (fabs(FN_angles[1] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 1;
					 }
					 else if (fabs(FN_angles[2] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 2;
					 }
					 else if (fabs(FN_angles[3] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 3;
					 }
					 else if (fabs(FN_angles[4] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 4;
					 }
					 else if (fabs(FN_angles[5] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 5;
					 }
					 else if (fabs(FN_angles[6] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 6;
					 }
					 else if (fabs(FN_angles[7] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 7;
					 }
					 else if (fabs(FN_angles[8] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 8;
					 }
					 else if (fabs(FN_angles[9] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 9;
					 }
					 else if (fabs(FN_angles[10] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 10;
					 }
					 else if (fabs(FN_angles[11] - angle) < epsilonRoundOff) {
						 NumOfNeighbor = 11;
					 }
				 }
			}
		}
	}

  // dtype getMatrixEntry(const unsigned i, const unsigned j) {
  //   pts2D ri = particles_X[i];//particles_X is a member of base class FMM_Matrix
  //   pts2D rj = particles_X[j];//particles_X is a member of base class FMM_Matrix
  //   double R2 = (ri.x-ri.y)*(ri.x-ri.y) + (rj.x-rj.y)*(rj.x-rj.y);
  //   double R = sqrt(R2);
  //   return exp(-I*R2);
  // }

	dtype getMatrixEntry(const unsigned i, const unsigned j) {
		pts2D ri = particles_X[i];//particles_X is a member of base class FMM_Matrix
		unsigned int leaf_i = i/rank;//identify to which leaf jth charge belongs
		unsigned int index_i = i%rank;
		unsigned int level_i = leafNodes[leaf_i].x;//level to which the leaf box that contains jth charge belongs
		unsigned int box_i = leafNodes[leaf_i].y;

		unsigned int leaf_j = j/rank;//identify to which leaf jth charge belongs
		unsigned int index_j = j%rank;
		unsigned int level_j = leafNodes[leaf_j].x;//level to which the leaf box that contains jth charge belongs
		unsigned int box_j = leafNodes[leaf_j].y;

		bool is_ij_neighbor = false;
		int typeOfNeighbor, NumOfNeighbor;
		getIfNeighbor(level_i, box_i, level_j, box_j, typeOfNeighbor, NumOfNeighbor);
		if (typeOfNeighbor != -1) {
			is_ij_neighbor = true;
		}

		if (is_ij_neighbor) {
			dtype entry;
			if (typeOfNeighbor == 0) { //colleagueNeighbors
				entry = colleagueNeighborInteraction[level_i-2][NumOfNeighbor](index_i, index_j);
			}
			else if (typeOfNeighbor == 1) { //coarseNeighbors
				entry = coarseNeighborInteraction[level_i-2][NumOfNeighbor](index_i, index_j);
			}
			else if (typeOfNeighbor == 2) { //fine neighbor
				entry = fineNeighborInteraction[level_i-2][NumOfNeighbor](index_i, index_j);
			}
			else { // separatedFineNeighbor
				entry = separatedFineNeighborInteraction[level_i-2][NumOfNeighbor](index_i, index_j);
			}
			if (findPhi) {
				entry = entry*kappa*kappa*(ContrastFunction(ri));
				if (i ==j) {
					entry = 1.0 + entry;
				}
			}
      if(abs(entry) < 1e-16) {
        entry = 0.0;
      }
			return entry;
		}
		else {
			double polar_angle_i = std::atan2(ri.y-tree[level_j][box_j].center.y, ri.x-tree[level_j][box_j].center.x);
			double R2	=	(ri.y-tree[level_j][box_j].center.y)*(ri.y-tree[level_j][box_j].center.y) + (ri.x-tree[level_j][box_j].center.x)*(ri.x-tree[level_j][box_j].center.x);
			double R	=	kappa*sqrt(R2);
			dtype tempSum = 0.0+0.0*I;
			int seriesLength = FF->seriesLength;
			#pragma omp parallel for
			for (int n = -seriesLength; n <= seriesLength; n++) {
				dtype temp = (besselJ(n, R) + I*besselY(n, R)) * exp(I*double(n)*polar_angle_i);
				tempSum += M[level_j](n+seriesLength, index_j)*temp;
			}
			tempSum *= I/4.0;
			if (findPhi) {
				tempSum *= kappa*kappa*(ContrastFunction(ri));
				if (i==j) {
					tempSum += 1.0;
				}
			}
      if(abs(tempSum) < 1e-16) {
        tempSum = 0.0;
      }
			return tempSum;
		}
	}

	void shift_scale_Nodes(std::vector<pts2D>& Nodes, std::vector<pts2D>& shifted_scaled_Nodes, double xShift, double yShift, double radius) {
		for (int k=0; k < Nodes.size(); ++k) {
			pts2D temp;
			temp.x	=	radius*Nodes[k].x+xShift;
			temp.y	=	radius*Nodes[k].y+yShift;
			shifted_scaled_Nodes.push_back(temp);
		}
	}

	//	get_ChebPoly
	double get_ChebPoly(double x, int n) {
		return cos(n*acos(x));
	}

	//	get_S
	double get_S(double x, double y, int n) {
		double S	=	0.5;
		#pragma omp parallel for
		for (int k=1; k<n; ++k) {
			S+=get_ChebPoly(x,k)*get_ChebPoly(y,k);
		}
		return 2.0/n*S;
	}
	//	set_Standard_Cheb_Nodes
	void set_Standard_Cheb_Nodes() {
		for (int k=0; k<nChebNodes; ++k) {
			standardChebNodes1D.push_back(-cos((k+0.5)/nChebNodes*PI));
		}
		pts2D temp1;
		for (int j=0; j<nChebNodes; ++j) {
			for (int k=0; k<nChebNodes; ++k) {
				temp1.x	=	standardChebNodes1D[k];
				temp1.y	=	standardChebNodes1D[j];
				standardChebNodes.push_back(temp1);
			}
		}
		//	Left Bottom child, i.e., Child 0
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x-0.5;
				temp1.y	=	0.5*temp1.y-0.5;
				standardChebNodesChild.push_back(temp1);
		}
		//	Right Bottom child, i.e., Child 1
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x+0.5;
				temp1.y	=	0.5*temp1.y-0.5;
				standardChebNodesChild.push_back(temp1);
		}
		//	Right Top child, i.e., Child 2
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x+0.5;
				temp1.y	=	0.5*temp1.y+0.5;
				standardChebNodesChild.push_back(temp1);
		}
		//	Left Top child, i.e., Child 3
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x-0.5;
				temp1.y	=	0.5*temp1.y+0.5;
				standardChebNodesChild.push_back(temp1);
		}
	}

	void get_Transfer_Matrix() {
		#pragma omp parallel for
		for (int l=0; l<4; ++l) {
			L2L[l]	=	Eigen::MatrixXd(rank,rank);
			#pragma omp parallel for
			for (int j=0; j<rank; ++j) {
				#pragma omp parallel for
				for (int k=0; k<rank; ++k) {
					L2L[l](j,k)	=	get_S(standardChebNodes[k].x, standardChebNodesChild[j+l*rank].x, nChebNodes)*get_S(standardChebNodes[k].y, standardChebNodesChild[j+l*rank].y, nChebNodes);
				}
			}
		}
		#pragma omp parallel for
		for (int l=0; l<4; ++l) {
			M2M[l]	=	L2L[l].transpose();
		}
	}
	//////////////////TREE CREATION////////////////////////////////////////////
	double resolveContrast(int j, int k) {
		Eigen::VectorXd contrastAtChebNodes;
		std::vector<pts2D> Nodes;
		shift_scale_Nodes(standardChebNodes, Nodes, tree[j][k].center.x, tree[j][k].center.y, tree[j][k].radius);
		EvaluateContrastAtNodes(contrastAtChebNodes, Nodes);
		Eigen::VectorXd approximateContrastAtChildChebNodes(4*rank);
		#pragma omp parallel for
		for (size_t c = 0; c < 4; c++) {
			approximateContrastAtChildChebNodes.segment(rank*c, rank) = L2L[c]*contrastAtChebNodes;
		}
		std::vector<pts2D> NodesChild;
		shift_scale_Nodes(standardChebNodesChild, NodesChild, tree[j][k].center.x, tree[j][k].center.y, tree[j][k].radius);
		Eigen::VectorXd contrastAtChildChebNodes;
		EvaluateContrastAtNodes(contrastAtChildChebNodes, NodesChild);
		double err = errInApproximation(contrastAtChildChebNodes, approximateContrastAtChildChebNodes);
		return err;
	}

		double resolveRHS(int j, int k) {
			Vec RHSAtChebNodes;
			std::vector<pts2D> Nodes;
			shift_scale_Nodes(standardChebNodes, Nodes, tree[j][k].center.x, tree[j][k].center.y, tree[j][k].radius);
			EvaluateRHSAtNodes(RHSAtChebNodes, Nodes);
			Vec approximateRHSAtChildChebNodes(4*rank);
			#pragma omp parallel for
			for (size_t c = 0; c < 4; c++) {
				approximateRHSAtChildChebNodes.segment(rank*c, rank) = L2L[c]*RHSAtChebNodes;
			}
			std::vector<pts2D> NodesChild;
			shift_scale_Nodes(standardChebNodesChild, NodesChild, tree[j][k].center.x, tree[j][k].center.y, tree[j][k].radius);
			Vec RHSAtChildChebNodes;
			EvaluateRHSAtNodes(RHSAtChildChebNodes, NodesChild);
			double err = errInApproximation(RHSAtChildChebNodes, approximateRHSAtChildChebNodes);
			return err;
		}

		void createTree(int j, int k) {
			//k: index of box in the vector tree[j]
			//b: boxNumber of box, tree[j][k]
			double errContrast = resolveContrast(j, k);
			double errRHS = resolveRHS(j, k);
			bool condition1Leaf = false;
			bool condition2Leaf = true; //need to be false
			bool condition3Leaf = false;
			if (errContrast < epsilonTree) condition1Leaf = true;
			if (errRHS < epsilonTree) condition2Leaf = true;
			if (kappa*tree[j][k].radius/2/PI <= 1.0) condition3Leaf = true;
			if (condition1Leaf && condition2Leaf && condition3Leaf) {
				tree[j][k].isLeaf = true;
				orderedPair op;
				op.x = j;
				op.y = k;
				leafNodes.push_back(op);
			}
			else {
				if (int(tree.size()) == j+1) {
					std::vector<LS2DBox> level;
					tree.push_back(level);
					std::vector<int> index;
				  indexTree.push_back(index);
				}
				int n	=	tree[j+1].size();
				int b = tree[j][k].boxNumber;
				for (size_t c = 0; c < 4; c++) {
					LS2DBox box;
					box.level = j+1;
					box.boxNumber		=	b*4+c;
					box.parentNumber	=	b;
					box.radius = 0.5*tree[j][k].radius;
					if (c==0) {
						box.center.x		=	tree[j][k].center.x-0.5*tree[j][k].radius;
						box.center.y		=	tree[j][k].center.y-0.5*tree[j][k].radius;
					}
					else if (c==1) {
						box.center.x		=	tree[j][k].center.x+0.5*tree[j][k].radius;
						box.center.y		=	tree[j][k].center.y-0.5*tree[j][k].radius;
					}
					else if (c==2) {
						box.center.x		=	tree[j][k].center.x+0.5*tree[j][k].radius;
						box.center.y		=	tree[j][k].center.y+0.5*tree[j][k].radius;
					}
					else {
						box.center.x		=	tree[j][k].center.x-0.5*tree[j][k].radius;
						box.center.y		=	tree[j][k].center.y+0.5*tree[j][k].radius;
					}
					tree[j+1].push_back(box);
					indexTree[j+1].push_back(box.boxNumber);
				}
				for (size_t c = 0; c < 4; c++) {
					createTree(j+1, n+c);
				}
			}
		}

		void createAdaptiveTree() {
			LS2DBox root;
			root.level = 0;
			root.boxNumber		=	0;
			root.parentNumber	=	-1;
			root.radius = L;
			root.center.x = 0.0;
			root.center.y = 0.0;
			std::vector<LS2DBox> level;
			level.push_back(root);
			tree.push_back(level);

			std::vector<int> index;
			index.push_back(0);
			indexTree.push_back(index);
			createTree(0, 0);
			nLevels = tree.size() - 1;
			// cout << "nLevels: " << nLevels << endl;
			for (size_t j = 1; j <= 10; j++) {
				boxRadius.push_back(boxRadius[j-1]/2.0);
			}
		}

    void check() {
      for (size_t j = 2; j <= 3; j++) {
        for (size_t k = 0; k < nBoxesPerLevel[j]; k++) {
          double errContrast = resolveContrast(j, k);
    			double errRHS = resolveRHS(j, k);
          cout << "j: " << j << "  k: " << k << endl;
          cout << "errContrast: " << errContrast << endl;
          cout << "errRHS: " << errRHS << endl;
        }
      }
    }

		void findnLevelsUniform(int j, int k) {
			//k: index of box in the vector tree[j]
			//b: boxNumber of box, tree[j][k]
			double errContrast = resolveContrast(j, k);
			double errRHS = resolveRHS(j, k);
			bool condition1Leaf = false;
			bool condition2Leaf = false;
			bool condition3Leaf = false;
      // cout << "errContrast: " << errContrast << endl;
      // cout << "errRHS: " << errRHS << endl;
			if (errContrast < epsilonTree) condition1Leaf = true;
			if (errRHS < epsilonTree) condition2Leaf = true;
			if (kappa*tree[j][k].radius/2/PI <= 1.0) condition3Leaf = true;
      if (condition1Leaf && condition2Leaf && condition3Leaf) {
			}
			else {
				if (int(tree.size()) == j+1) {
					std::vector<LS2DBox> level;
					tree.push_back(level);
				}
				int n	=	tree[j+1].size();
				int b = tree[j][k].boxNumber;
				for (size_t c = 0; c < 4; c++) {
					LS2DBox box;
					box.level = j+1;
					box.boxNumber		=	b*4+c;
					box.parentNumber	=	b;
					box.radius = 0.5*tree[j][k].radius;
					if (c==0) {
						box.center.x		=	tree[j][k].center.x-0.5*tree[j][k].radius;
						box.center.y		=	tree[j][k].center.y-0.5*tree[j][k].radius;
					}
					else if (c==1) {
						box.center.x		=	tree[j][k].center.x+0.5*tree[j][k].radius;
						box.center.y		=	tree[j][k].center.y-0.5*tree[j][k].radius;
					}
					else if (c==2) {
						box.center.x		=	tree[j][k].center.x+0.5*tree[j][k].radius;
						box.center.y		=	tree[j][k].center.y+0.5*tree[j][k].radius;
					}
					else {
						box.center.x		=	tree[j][k].center.x-0.5*tree[j][k].radius;
						box.center.y		=	tree[j][k].center.y+0.5*tree[j][k].radius;
					}
					tree[j+1].push_back(box);
				}
				for (size_t c = 0; c < 4; c++) {
					findnLevelsUniform(j+1, n+c);
				}
			}
		}

    void createTreeU(int j, int k) {
      //k: index of box in the vector tree[j]
      //b: boxNumber of box, tree[j][k]
      if (j == nLevels) {
      // if (j == nLevelsUniform) {
        tree[j][k].isLeaf = true;
        orderedPair op;
        op.x = j;
        op.y = k;
        leafNodes.push_back(op);
      }
      else {
        if (int(tree.size()) == j+1) {
          std::vector<LS2DBox> level;
          tree.push_back(level);
          std::vector<int> index;
          indexTree.push_back(index);
        }
        int n	=	tree[j+1].size();
        int b = tree[j][k].boxNumber;
        for (size_t c = 0; c < 4; c++) {
          LS2DBox box;
          box.level = j+1;
          box.boxNumber		=	b*4+c;
          box.parentNumber	=	b;
          box.radius = 0.5*tree[j][k].radius;
          if (c==0) {
            box.center.x		=	tree[j][k].center.x-0.5*tree[j][k].radius;
            box.center.y		=	tree[j][k].center.y-0.5*tree[j][k].radius;
          }
          else if (c==1) {
            box.center.x		=	tree[j][k].center.x+0.5*tree[j][k].radius;
            box.center.y		=	tree[j][k].center.y-0.5*tree[j][k].radius;
          }
          else if (c==2) {
            box.center.x		=	tree[j][k].center.x+0.5*tree[j][k].radius;
            box.center.y		=	tree[j][k].center.y+0.5*tree[j][k].radius;
          }
          else {
            box.center.x		=	tree[j][k].center.x-0.5*tree[j][k].radius;
            box.center.y		=	tree[j][k].center.y+0.5*tree[j][k].radius;
          }
          tree[j+1].push_back(box);
          indexTree[j+1].push_back(box.boxNumber);
        }
        for (size_t c = 0; c < 4; c++) {
          createTreeU(j+1, n+c);
        }
      }
    }

		void createUniformTree() {
			LS2DBox root;
			root.level = 0;
			root.boxNumber		=	0;
			root.parentNumber	=	-1;
			root.radius = L;
			root.center.x = 0.0;
			root.center.y = 0.0;
			std::vector<LS2DBox> level;
			level.push_back(root);
			tree.push_back(level);

			std::vector<int> index;
			index.push_back(0);
			indexTree.push_back(index);

      // findnLevelsUniform(0, 0);
      // nLevels = tree.size();
      // std::cout << "LS nLevels: " << nLevels << std::endl;
      // tree.clear();
      // level.push_back(root);
			// tree.push_back(level);

      nLevels = nLevelsUniform;

      createTreeU(0, 0);
			nLevels = tree.size() - 1;
      for (size_t j = 1; j <= nLevels+1; j++) {// nLevels+1 needed in finding getOperator in neighbor interaction
        nBoxesPerLevel.push_back(nBoxesPerLevel[j-1]*4);
				boxRadius.push_back(boxRadius[j-1]/2.0);
			}
		}

		void getFutureGeneration_LFR(int bj, int bk, int pnj, int pni, std::vector<orderedPair>& futureGeneration) {
			if (tree[pnj][pni].isLeaf) {
				if ( fabs(tree[bj][bk].center.x - tree[pnj][pni].center.x) >= 3.0*boxRadius[bj] + boxRadius[pnj]-epsilonRoundOff
				 || fabs(tree[bj][bk].center.y - tree[pnj][pni].center.y) >= 3.0*boxRadius[bj] + boxRadius[pnj]-epsilonRoundOff) {
				}
				else {
					orderedPair op;
					op.x = pnj;
					op.y = pni;
					futureGeneration.push_back(op);
				}
			}
			else {
				for (int nc=0; nc<4; ++nc) { //children of parents neighbors
					int pnn = tree[pnj][pni].boxNumber;//boxNumber
					int boxB = 4*pnn+nc;//its index=?
					std::vector<int>::iterator indx = std::find(indexTree[pnj+1].begin(), indexTree[pnj+1].end(), boxB);
					int boxB_index = indx-indexTree[pnj+1].begin();
					getFutureGeneration_LFR(bj, bk, pnj+1, boxB_index, futureGeneration);
				}
			}
		}

		void assign_Child_Interaction_LFR(int c, int j, int k, std::vector<std::vector<std::vector<orderedPair> > >& Tree_Neighbors_LFR) {
			int parentboxNumber = tree[j][k].boxNumber;
			int boxA = 4*parentboxNumber+c;//child box number; its index=?
			std::vector<int>::iterator indx = std::find(indexTree[j+1].begin(), indexTree[j+1].end(), boxA);
			int boxA_index = indx-indexTree[j+1].begin();
			for (int n=0; n<Tree_Neighbors_LFR[j][k].size(); ++n) {//parents neighbors; so u need its index to access it which is k
				//children of neighbors of parent which are not neighbors to child=IL
				int pnj = Tree_Neighbors_LFR[j][k][n].x;//level
				int pni = Tree_Neighbors_LFR[j][k][n].y;//index
				int pnn = tree[pnj][pni].boxNumber;//boxNumber

				std::vector<orderedPair> futureGeneration;
				getFutureGeneration_LFR(j+1, boxA_index, pnj, pni, futureGeneration);
				for (size_t d = 0; d < futureGeneration.size(); d++) {
					Tree_Neighbors_LFR[j+1][boxA_index].push_back(futureGeneration[d]);
				}
			}
		}

		//	Assigns the interactions for the children of a box
		void assign_Box_Interactions_LFR(int j, int k, std::vector<std::vector<std::vector<orderedPair> > >& Tree_Neighbors_LFR) {
			if (!tree[j][k].isLeaf) {
				//#pragma omp parallel for
				for (int c=0; c<4; ++c) {
					assign_Child_Interaction_LFR(c,j,k, Tree_Neighbors_LFR);
				}
			}
		}

		//	Assigns the interactions for the children all boxes at a given level
		void assign_Level_Interactions_LFR(int j, std::vector<std::vector<std::vector<orderedPair> > >& Tree_Neighbors_LFR) {
			//#pragma omp parallel for
			for (int k=0; k<tree[j].size(); ++k) {
				assign_Box_Interactions_LFR(j,k, Tree_Neighbors_LFR);//k is index number of box in tree[j] vector
			}
		}

		//	Assigns the interactions for the children all boxes in the tree
		void assign_Tree_Interactions_LFR(std::vector<std::vector<std::vector<orderedPair> > >& Tree_Neighbors_LFR) {
			int J = 1;
			for (int c=0; c<4; ++c) {
				for (int n=0; n<4; ++n) {
					orderedPair op;
					op.x = J;
					op.y = n;
					Tree_Neighbors_LFR[J][c].push_back(op);
				}
			}
			for (int j=1; j<=nLevels-1; ++j) {
				assign_Level_Interactions_LFR(j, Tree_Neighbors_LFR);
			}
		}

		void makeLevelRestriction_Tree() {
			for (size_t n = 0; n < nLevels; n++) {
				std::vector<std::vector<std::vector<orderedPair> > > Tree_Neighbors_LFR;
				for (size_t j = 0; j <= nLevels; j++) {
					std::vector<std::vector<orderedPair> > level;
					for (size_t k = 0; k < tree[j].size(); k++) {
						std::vector<orderedPair> box;
						level.push_back(box);
					}
					Tree_Neighbors_LFR.push_back(level);
				}
				assign_Tree_Interactions_LFR(Tree_Neighbors_LFR);
				std::vector<orderedPair> leafNodesOld = leafNodes;
				for (size_t l = 0; l < leafNodesOld.size(); l++) {
					makeLevelRestriction_Box(l, leafNodes[l], Tree_Neighbors_LFR, leafNodesOld);
				}

				int cnt = 0;
 				for (size_t l = 0; l < leafNodesOld.size(); l++) {//removing all the non-leaves from leafNodes vector
					if (leafNodesOld[l].x == -1) {
						leafNodes.erase(leafNodes.begin()+l-cnt);
						cnt++;
					}
				}

				for (size_t j = 0; j <= nLevels; j++) {
					for (size_t k = 0; k < Tree_Neighbors_LFR[j].size(); k++) {
						Tree_Neighbors_LFR[j][k].clear();
					}
					Tree_Neighbors_LFR[j].clear();
				}
				Tree_Neighbors_LFR.clear();
			}
		}

		int getIndexOfInLeafNodes(orderedPair op) {
			for (size_t l = 0; l < leafNodes.size(); l++) {
				if (op.x == leafNodes[l].x && op.y == leafNodes[l].y) {
					return l;
				}
			}
		}

		void makeLevelRestriction_Box(int l, orderedPair leaf, std::vector<std::vector<std::vector<orderedPair> > >& Tree_Neighbors_LFR, std::vector<orderedPair>& leafNodesOld) {
			//l is index of leaf in leafNodes
			int j = leaf.x;
			int k = leaf.y;
			for (size_t i = 0; i < Tree_Neighbors_LFR[j][k].size(); i++) {
				int nj = Tree_Neighbors_LFR[j][k][i].x;
				int nk = Tree_Neighbors_LFR[j][k][i].y;
				std::vector<orderedPair> newLeaves;

				if (j-nj >= 2 || nj-j >= 2) {
					if (j-nj >= 2) {
						int indexOfBox = getIndexOfInLeafNodes(Tree_Neighbors_LFR[j][k][i]);//index of Tree_Neighbors_LFR[j][k][i]
						if (leafNodesOld[indexOfBox].x != -1) {
							refineBox(indexOfBox, Tree_Neighbors_LFR[j][k][i], leafNodesOld);
						}
					}
					else if(nj-j >= 2) {
						if (leafNodesOld[l].x != -1) {
							refineBox(l, leaf, leafNodesOld);
						}
					}
				}
			}
		}

		void refineBox(int l, orderedPair leaf, std::vector<orderedPair> &leafNodesOld) {
			int j = leaf.x; //level
			int k = leaf.y; //box index
			int b = tree[j][k].boxNumber;
			leafNodesOld[l].x = -1;//making it a non-leaf node
			tree[j][k].isLeaf = false;
			for (size_t c = 0; c < 4; c++) {
				LS2DBox box;
				box.isLeaf = true;
				box.level = j+1;
				box.boxNumber		=	b*4+c;
				box.parentNumber	=	b;
				box.radius = 0.5*tree[j][k].radius;
				if (c==0) {
					box.center.x		=	tree[j][k].center.x-0.5*tree[j][k].radius;
					box.center.y		=	tree[j][k].center.y-0.5*tree[j][k].radius;
				}
				else if (c==1) {
					box.center.x		=	tree[j][k].center.x+0.5*tree[j][k].radius;
					box.center.y		=	tree[j][k].center.y-0.5*tree[j][k].radius;
				}
				else if (c==2) {
					box.center.x		=	tree[j][k].center.x+0.5*tree[j][k].radius;
					box.center.y		=	tree[j][k].center.y+0.5*tree[j][k].radius;
				}
				else {
					box.center.x		=	tree[j][k].center.x-0.5*tree[j][k].radius;
					box.center.y		=	tree[j][k].center.y+0.5*tree[j][k].radius;
				}
				orderedPair op;
				op.x = j+1;
				op.y = tree[j+1].size();
				leafNodes.push_back(op);
				tree[j+1].push_back(box);
				indexTree[j+1].push_back(box.boxNumber);
			}
		}

		void assignLeafChargeLocations(std::vector<pts2D> &particles_out) {
			for (size_t k = 0; k < leafNodes.size(); k++) {
				int j = leafNodes[k].x;
				int b = leafNodes[k].y;
				int startIndex = gridPoints.size();
				for (size_t i = 0; i < rank; i++) {
					tree[j][b].chargeLocations.push_back(startIndex+i);
				}
				shift_scale_Nodes(standardChebNodes, gridPoints, tree[j][b].center.x, tree[j][b].center.y, boxRadius[j]);
			}
			particles_X = gridPoints;//object of base class FMM_Matrix
			particles_Y = gridPoints;
			particles_out = gridPoints;
		}

		void assignNonLeafChargeLocations() {
			for (int j=nLevels-1; j>1; --j) {
				int J	=	j+1;
				//#pragma omp parallel for
				for (int k=0; k<tree[j].size(); ++k) {
					if (!tree[j][k].isLeaf) {
						int b = tree[j][k].boxNumber;
						int KboxNumber;
						int K[4];
						std::vector<int>::iterator indx;
						KboxNumber = 4*b+0;
						indx = std::find(indexTree[J].begin(), indexTree[J].end(), KboxNumber);
						K[0] = indx-indexTree[J].begin();
						KboxNumber = 4*b+1;
						indx = std::find(indexTree[J].begin(), indexTree[J].end(), KboxNumber);
						K[1] = indx-indexTree[J].begin();
						KboxNumber = 4*b+2;
						indx = std::find(indexTree[J].begin(), indexTree[J].end(), KboxNumber);
						K[2] = indx-indexTree[J].begin();
						KboxNumber = 4*b+3;
						indx = std::find(indexTree[J].begin(), indexTree[J].end(), KboxNumber);
						K[3] = indx-indexTree[J].begin();

						for (int c=0; c<4; ++c) {
							//Block containing n elements, starting at position i: vector.segment(i,n)
							for (int i = 0; i < tree[J][K[c]].chargeLocations.size(); i++) {
								tree[j][k].chargeLocations.push_back(tree[J][K[c]].chargeLocations[i]);
							}
						}
					}
				}
			}
		}

		void assignLeafChebNodes() {
			#pragma omp parallel for
			for (size_t k = 0; k < leafNodes.size(); k++) {
				int j = leafNodes[k].x;
				int b = leafNodes[k].y;
				shift_scale_Nodes(standardChebNodes, tree[j][b].chebNodes, tree[j][b].center.x, tree[j][b].center.y, boxRadius[j]);
			}
		}

		double errInApproximation(Eigen::VectorXd& trueValue, Eigen::VectorXd& approximateValue) {
			Eigen::VectorXd error(trueValue.size());
			#pragma omp parallel for
			for (int k=0; k<trueValue.size(); ++k) {
				error(k)	=	fabs(trueValue(k)-approximateValue(k));
				//error(k)	=	fabs((trueValue-approximateValue)(k)/trueValue(k));
			}
			return error.maxCoeff();///trueValue.maxCoeff();
		}

		double errInApproximation(Vec& trueValue, Vec& approximateValue) {
			Vec error;
			error = trueValue - approximateValue;
			VectorXd errorAbs = error.cwiseAbs();
			VectorXd trueValueAbs = trueValue.cwiseAbs();
			return errorAbs.maxCoeff();///trueValueAbs.maxCoeff();
		}

		void EvaluateContrastAtNodes(Eigen::VectorXd& contrastAtChebNodes, std::vector<pts2D>& Nodes) {
			contrastAtChebNodes = Eigen::VectorXd(Nodes.size());
			#pragma omp parallel for
			for (size_t i = 0; i < Nodes.size(); i++) {
				contrastAtChebNodes(i) = ContrastFunction(Nodes[i]);
			}
		}

		void EvaluateRHSAtNodes(Vec& RHSAtChebNodes, std::vector<pts2D>& Nodes) {
			RHSAtChebNodes = Vec(Nodes.size());
			#pragma omp parallel for
			for (size_t i = 0; i < Nodes.size(); i++) {
				RHSAtChebNodes(i) = RHSFunction(Nodes[i]);
			}
		}

		void writeMToFile() {
			//create directory
			char final [512];
			MFilename = "M";
		  sprintf (final, "%s/precomputations/%s",currentDirectory.c_str(), MFilename.c_str());
			mkdir(final, 0775);

      std::string filename;
			MFilename = "M/M_" + std::to_string(nChebNodes) + "_" + std::to_string(int(kappa)) + "_" + std::to_string(degreeOfBases);
			sprintf (final, "%s/precomputations/%s",currentDirectory.c_str(), MFilename.c_str());
			mkdir(final, 0775);
			filename = string(final) + "/M_" + std::to_string(nChebNodes);
			for (size_t l = 0; l < M.size(); l++) {
				string filenameModified = filename + "_" + std::to_string(l);
				std::ofstream myfile;
				myfile.open(filenameModified.c_str());
				myfile << M[l] << endl;
				myfile.close();
			}
		}

		void ReadFromTextFile(Mat &matrix ,std::string filename) {
			std::ifstream inFile (filename,std::ios::in);
			if(!inFile.good()) {
				std::cout<<"Error: could not open file:\""<<filename<<"\"for reading \n";
				exit (2);
			}
			//find the no of values in file
			std::istream_iterator<std::string> in{inFile};
			std::istream_iterator<std::string> end;
			long numberofWords=std::distance(in,end);
			//find the no of lines in file
			inFile.clear();
			inFile.seekg(0,std::ios::beg);
			long numberofLines=std::count(std::istreambuf_iterator<char>(inFile),std::istreambuf_iterator<char>(),'\n');
			long rows=numberofLines;
			long cols=numberofWords/numberofLines;
			if(rows*cols!=numberofWords) {
				std::cout<<"\n Infile"<<filename<<"cannot form a matrix \n";
				exit(2);
			}
			matrix.array().resize(rows,cols);
			//matrix Base does not allow resizing ...hence change array base
			inFile.clear();
			inFile.seekg(0,std::ios::beg);
			for(unsigned int i=0;i<matrix.rows();i++)
			 for(unsigned int j=0;j<matrix.cols();j++)
				 inFile>>matrix(i,j);
			inFile.close();
		}

		void readMFromFile() {
      std::string filename;
			std::vector<Mat> M2;
			MFilename = currentDirectory + "/precomputations" + "/M/M_" + std::to_string(nChebNodes) + "_" + std::to_string(int(kappa)) + "_" + std::to_string(degreeOfBases);
			filename = MFilename + "/M_" + std::to_string(nChebNodes);
			for (size_t l = 0; l <= nLevels; l++) {
				string filenameModified = filename + "_" + std::to_string(l);
				Mat m;
				ReadFromTextFile(m ,filenameModified);
				M.push_back(m);
			}
		}

		void outputAdaptiveGrid(std::string filename) {
			double xcenter = 0.0;
			double ycenter = 0.0;
			double Lx = L;
			double Ly = L;

			std::ofstream myfile;
			myfile.open(filename.c_str());
			myfile << "\\documentclass{standalone}" << std::endl;
			myfile << "\\usepackage{tikz}" << std::endl;
			myfile << "\\begin{document}" << std::endl;
			myfile << "\\begin{tikzpicture}" << std::endl;
			for (int k=0; k<(int)leafNodes.size(); ++k) {
				int j = leafNodes[k].x;
				int b = leafNodes[k].y;
				myfile << "\\draw (" << tree[j][b].center.x-tree[j][b].radius << ",";
				myfile << tree[j][b].center.y-tree[j][b].radius << ") rectangle (";
				//myfile << tree[j][b].center.y-tree[j][b].radius << ") rectangle node{\\tiny " << b << "} (";
				//myfile << tree[j][b].center.y-tree[j][b].radius << ") rectangle node{\\tiny " << tree[j][b].boxNumber << "} (";
				myfile << tree[j][b].center.x+tree[j][b].radius << ",";
				myfile << tree[j][b].center.y+tree[j][b].radius << ");" << std::endl;
			}
			long double push	=	0.125;
			myfile<< "\\node at (" << xcenter-Lx-push << "," << ycenter-Ly-push << ") {\\tiny$(" << xcenter-Lx << "," << ycenter-Ly << ")$};" << std::endl;
			myfile<< "\\node at (" << xcenter-Lx-push << "," << ycenter+Ly+push << ") {\\tiny$(" << xcenter-Lx << "," << ycenter+Ly << ")$};" << std::endl;
			myfile<< "\\node at (" << xcenter+Lx+push << "," << ycenter-Ly-push << ") {\\tiny$(" << xcenter+Lx << "," << ycenter-Ly << ")$};" << std::endl;
			myfile<< "\\node at (" << xcenter+Lx+push << "," << ycenter+Ly+push << ") {\\tiny$(" << xcenter+Lx << "," << ycenter+Ly << ")$};" << std::endl;
			myfile << "\\end{tikzpicture}" << std::endl;
			myfile << "\\end{document}" << std::endl;
			myfile.close();
		}
		//////////////////////////////////////////////////////////////

	double find_distance(int j1, int k1, int j2, int k2) {
		pts2D r1 = tree[j1][k1].center;
		pts2D r2 = tree[j2][k2].center;
		return	sqrt((r1.x-r2.x)*(r1.x-r2.x) + (r1.y-r2.y)*(r1.y-r2.y));
	}

	void getNeighbors() {//with respect to unit box at origin; this is assumed to leaf
		//for a leaf at level_A; the centers need to scaled up by boxRadius[level_A]
		#pragma omp parallel for
		for (size_t i = 0; i < 20; i++) {
			SFN_angles[i] = atan2(SFN_centers[i].y, SFN_centers[i].x);
		}
		#pragma omp parallel for
		for (size_t i = 0; i < 9; i++) {
			N_angles[i] = atan2(N_centers[i].y, N_centers[i].x);
		}
		#pragma omp parallel for
		for (size_t i = 0; i < 12; i++) {
			FN_angles[i] = atan2(FN_centers[i].y, FN_centers[i].x);
		}
		#pragma omp parallel for
		for (size_t i = 0; i < 12; i++) {
			CN_angles[i] = atan2(CN_centers[i].y, CN_centers[i].x);
		}
	}

	//the following function is for Lippmann Schwinger problem(uses equations specific for neighbor interactions)
	void getOperator(int level_A, int level_B, pts2D center_B, pts2D center_A, Mat &matOperator) {
		integrandInputs II;
		II.center_B = center_B;
		II.center_A = center_A;
		II.level_A = level_A;
		II.level_B = level_B;//colleague Neighbor; so same level
		II.radius_A = boxRadius[level_A];
		II.radius_B = boxRadius[level_B];
		NearFieldInteraction1 *NFI1 = new NearFieldInteraction1(nChebNodes, II, degreeOfBases);
		NearFieldInteraction2 *NFI2 = new NearFieldInteraction2(nChebNodes, II, degreeOfBases);
		Mat V1;
		Mat V2;

		//the following two lines use no expansion of H(0,x); it uses H(0,x)=J(0,x)+iY(0,x)
		NFI1->evaluateV(V1);
		Mat temp = V1;

		//the following lines use series expansion of H(0,x); it uses series expansion of J(0,x) and Y(0,x)
		//NFI1->evaluateV1(V1);
		//NFI2->evaluateV2(V2);
		//Mat temp = V1+V2;


		Mat V_temp = temp*I*boxRadius[level_B]*boxRadius[level_B]/4.0;
		matOperator = V_temp*Q_pinv;
		delete NFI1;
		delete NFI2;
	}


	void getNeighborInteractions() {
		//assume domain is always [-1,1]^{2}
		#pragma omp parallel for
		for (size_t level_A = 2; level_A <= nLevels; level_A++) {
			writeNeighborInteractionsForLevel(level_A);
		}
	}

	void writeNeighborInteractionsForLevel(int level_A) {
		Mat colleagueNeighborInteractionTemp[9];
		Mat fineNeighborInteractionTemp[12];
		Mat separatedFineNeighborInteractionTemp[20];
		Mat coarseNeighborInteractionTemp[12];
		pts2D center_A;
		center_A.x = 0.0;
		center_A.y = 0.0;
		#pragma omp parallel for
		for (size_t i = 0; i < 9; i++) { //colleagueNeighbors
			int level_B = level_A;
			pts2D center_B = N_centers[i];
			center_B.x *= boxRadius[level_A];
			center_B.y *= boxRadius[level_A];
			getOperator(level_A, level_B, center_B, center_A, colleagueNeighborInteractionTemp[i]);
		}
		#pragma omp parallel for
		for (size_t i = 0; i < 12; i++) { //coarseNeighbors
			int level_B = level_A-1;
			pts2D center_B = CN_centers[i];
			center_B.x *= boxRadius[level_A];
			center_B.y *= boxRadius[level_A];
			getOperator(level_A, level_B, center_B, center_A, coarseNeighborInteractionTemp[i]);
		}
		#pragma omp parallel for
		for (size_t i = 0; i < 20; i++) { //separatedFineNeighbors
			int level_B = level_A+1;
			pts2D center_B = SFN_centers[i];
			center_B.x *= boxRadius[level_A];
			center_B.y *= boxRadius[level_A];
			getOperator(level_A, level_B, center_B, center_A, separatedFineNeighborInteractionTemp[i]);
		}
		#pragma omp parallel for
		for (size_t i = 0; i < 12; i++) { //fineNeighbors
			int level_B = level_A+1;
			pts2D center_B = FN_centers[i];
			center_B.x *= boxRadius[level_A];
			center_B.y *= boxRadius[level_A];
			getOperator(level_A, level_B, center_B, center_A, fineNeighborInteractionTemp[i]);
		}
		//create directory
		NeighborFilename = "Neighbor";
	  char final [256];
	  sprintf (final, "%s/precomputations/%s",currentDirectory.c_str(), NeighborFilename.c_str());
	  mkdir(final, 0775);

		NeighborFilename = "Neighbor/Neighbor_" + std::to_string(nChebNodes) + "_" + std::to_string(int(kappa)) + "_" + std::to_string(degreeOfBases);
		sprintf (final, "%s/precomputations/%s",currentDirectory.c_str(), NeighborFilename.c_str());
	  mkdir(final, 0775);

		//writeToFile
		std::string filename;
		filename = string(final) + "/colleagueNeighborInteraction";
		filename = filename + "_" + std::to_string(level_A);
		#pragma omp parallel for
		for (size_t n = 0; n < 9; n++) {
			string filenameModified = filename + "_" + std::to_string(n);
			std::ofstream myfile;
			myfile.open(filenameModified.c_str());
			myfile << colleagueNeighborInteractionTemp[n] << endl;
			myfile.close();
		}
		filename = string(final) + "/fineNeighborInteraction";
		filename = filename + "_" + std::to_string(level_A);
		#pragma omp parallel for
		for (size_t n = 0; n < 12; n++) {
			string filenameModified = filename + "_" + std::to_string(n);
			std::ofstream myfile;
			myfile.open(filenameModified.c_str());
			myfile << fineNeighborInteractionTemp[n] << endl;
			myfile.close();
		}
		filename = string(final) + "/separatedFineNeighborInteraction";
		filename = filename + "_" + std::to_string(level_A);
		#pragma omp parallel for
		for (size_t n = 0; n < 20; n++) {
			string filenameModified = filename + "_" + std::to_string(n);
			std::ofstream myfile;
			myfile.open(filenameModified.c_str());
			myfile << separatedFineNeighborInteractionTemp[n] << endl;
			myfile.close();
		}
		filename = string(final) + "/coarseNeighborInteraction";
		filename = filename + "_" + std::to_string(level_A);
		#pragma omp parallel for
		for (size_t n = 0; n < 12; n++) {
			string filenameModified = filename + "_" + std::to_string(n);
			std::ofstream myfile;
			myfile.open(filenameModified.c_str());
			myfile << coarseNeighborInteractionTemp[n] << endl;
			myfile.close();
		}
	}

	void readNeighborInteractions() {
		int numberOfLevelOperators = nLevels-2+1;
		std::string filename;
		NeighborFilename = currentDirectory + "/precomputations" + "/Neighbor/Neighbor_" + std::to_string(nChebNodes) + "_" + std::to_string(int(kappa)) + "_" + std::to_string(degreeOfBases);
		for (size_t level_A = 2; level_A <= nLevels; level_A++) {
			filename = NeighborFilename + "/colleagueNeighborInteraction";
			filename = filename + "_" + std::to_string(level_A);
			#pragma omp parallel for
			for (size_t n = 0; n < 9; n++) {
				string filenameModified = filename + "_" + std::to_string(n);
				ReadFromTextFile(colleagueNeighborInteraction[level_A-2][n] ,filenameModified);
			}

			filename = NeighborFilename + "/fineNeighborInteraction";
			filename = filename + "_" + std::to_string(level_A);
			#pragma omp parallel for
			for (size_t n = 0; n < 12; n++) {
				string filenameModified = filename + "_" + std::to_string(n);
				ReadFromTextFile(fineNeighborInteraction[level_A-2][n] ,filenameModified);
			}

			filename = NeighborFilename + "/separatedFineNeighborInteraction";
			filename = filename + "_" + std::to_string(level_A);
			#pragma omp parallel for
			for (size_t n = 0; n < 20; n++) {
				string filenameModified = filename + "_" + std::to_string(n);
				ReadFromTextFile(separatedFineNeighborInteraction[level_A-2][n] ,filenameModified);
			}

			filename = NeighborFilename + "/coarseNeighborInteraction";
			filename = filename + "_" + std::to_string(level_A);
			#pragma omp parallel for
			for (size_t n = 0; n < 12; n++) {
				string filenameModified = filename + "_" + std::to_string(n);
				ReadFromTextFile(coarseNeighborInteraction[level_A-2][n] ,filenameModified);
			}
		}
	}

};
#endif
