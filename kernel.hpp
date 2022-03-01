//
//  kernel.hpp
//
//
//  Created by Vaishnavi Gujjula on 1/4/21.
//
//
#ifndef __kernel_hpp__
#define __kernel_hpp__

#include "LS/LS_Assembly.hpp" //Lippmann-Schwinger kernel
#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
// #define EIGEN_DONT_PARALLELIZE
using namespace Eigen;

// const double PI	=	3.1415926535897932384;
#include <map>
// struct pts2D {
// 	double x,y;
// };

#ifdef USE_COMPLEX64
    using dtype=std::complex<double>;
    using dtype_base=double;
    using Mat=Eigen::MatrixXcd;
    using Vec=Eigen::VectorXcd;
    // const std::complex<double> I(0.0, 1.0);
#endif

class kernel {
public:
  bool isTrans;		//	Checks if the kernel is translation invariant, i.e., the kernel is K(r).
	bool isHomog;		//	Checks if the kernel is homogeneous, i.e., K(r) = r^{alpha}.
	bool isLogHomog;	//	Checks if the kernel is log-homogeneous, i.e., K(r) = log(r^{alpha}).
	double alpha;		//	Degree of homogeneity of the kernel.
  double a;

  std::vector<pts2D> particles_X;
	std::vector<pts2D> particles_Y;

	kernel(std::vector<pts2D>& particles_X, std::vector<pts2D>& particles_Y) {
			this->particles_X = particles_X;
			this->particles_Y = particles_Y;
	}

	virtual dtype getMatrixEntry(const unsigned i, const unsigned j) {
		std::cout << "virtual getInteraction" << std::endl;
		return 0.0+0.0*I;
	}

	Vec getRow(const int j, std::vector<int> col_indices) {
		int n_cols = col_indices.size();
		Vec row(n_cols);
    #pragma omp parallel for
    for(int k = 0; k < n_cols; k++) {
        row(k) = this->getMatrixEntry(j, col_indices[k]);
    }
    return row;
  }

  Vec getCol(const int k, std::vector<int> row_indices) {
		int n_rows = row_indices.size();
    Vec col(n_rows);
    #pragma omp parallel for
    for (int j=0; j<n_rows; ++j) {
			col(j) = this->getMatrixEntry(row_indices[j], k);
    }
    return col;
  }

  Mat getMatrix(std::vector<int> row_indices, std::vector<int> col_indices) {
		int n_rows = row_indices.size();
		int n_cols = col_indices.size();
    Mat mat(n_rows, n_cols);
    #pragma omp parallel for
    for (int j=0; j < n_rows; ++j) {
        #pragma omp parallel for
        for (int k=0; k < n_cols; ++k) {
            mat(j,k) = this->getMatrixEntry(row_indices[j], col_indices[k]);
        }
    }
    return mat;
  }
  ~kernel() {};
};

class userkernel: public kernel {
private:
  LS2DTree* F;
  int nCones_LFR;//number of cones to be used in DAFMM
  double TOL_POW;//currently not in use
  int nLevelsUniform; //number of levels to be constructed in case of uniform tree
  int yes2DFMM; //DFMM or FMM

public:
  Eigen::MatrixXcd Aexplicit;
  std::vector<pts2D> particles_X;//dummy
  std::vector<pts2D> particles_Y;//dummy
  int N;
  std::vector<pts2D> gridPoints; // location of particles in the domain
  Vec rhs;
  int nLevelsResolve;
  userkernel(int nChebNodes, int treeAdaptivity, int degreeOfBases, double L, std::vector<pts2D>& particles_X, std::vector<pts2D>& particles_Y, int nLevelsUniform) : kernel(particles_X, particles_Y) {
    isTrans		=	true;
    isHomog		=	true;
    isLogHomog	=	false;
    alpha		=	-1.0;

    this->nCones_LFR			=	16;//number of cones to be used in DAFMM
    this->TOL_POW         = 9;//currently not in use
    this->nLevelsUniform  = nLevelsUniform; //number of levels to be constructed in case of uniform tree
    this->yes2DFMM				=	0; //DFMM or FMM
    this->F	=	new LS2DTree(nCones_LFR, nChebNodes, L, yes2DFMM, TOL_POW, particles_X, particles_Y, kappa, degreeOfBases, treeAdaptivity, nLevelsUniform);
    this->gridPoints = F->gridPoints;
    this->nLevelsResolve = F->nLevels;
    this->N = gridPoints.size(); // locations of particles in the domain
    // std::cout << "N: " << N << std::endl;
    // std::cout << "gridPoints.S: " << gridPoints.size() << std::endl;
    rhs = Vec(N);
    for (size_t i = 0; i < this->N; i++) {
      this->rhs(i) = F->RHSFunction(gridPoints[i]);
    }
    Aexplicit = Eigen::MatrixXcd::Random(1024,1024);
    // Aexplicit = Eigen::MatrixXcd::Random(1008,1008);
  }

  dtype getMatrixEntry(const unsigned i, const unsigned j) {
    dtype output;
    output = F->getMatrixEntry(i, j);
    // return Aexplicit(i, j);
    return output;
    // return output.real()+I*output.imag();
  }

  // dtype getMatrixEntry(const unsigned i, const unsigned j) {
  //   return(Aexplicit(i,j));
  //   // return(Aexplicit(i,j)+Aexplicit(j,i));
  // }


	// dtype getMatrixEntry(const unsigned i, const unsigned j) {
	// 	pts2D r1 = particles_X[i];//particles_X is a member of base class FMM_Matrix
	// 	pts2D r2 = particles_X[j];//particles_X is a member of base class FMM_Matrix
	// 	double R2	=	(r1.x-r2.x)*(r1.x-r2.x) + (r1.y-r2.y)*(r1.y-r2.y);
  //   double R = sqrt(R2);
	// 	// if (R < 1e-10) {
	// 	// 	return 1.0;
	// 	// }
	// 	if (R < a) {
	// 		return R/a+1.0;
	// 	}
	// 	else {
	// 		return a/R;
	// 	}
	// }

  // dtype getMatrixEntry(const unsigned i, const unsigned j) {
	// 	pts2D r1 = particles_X[i];//particles_X is a member of base class FMM_Matrix
	// 	pts2D r2 = particles_X[j];//particles_X is a member of base class FMM_Matrix
	// 	double R2	=	(r1.x-r2.x)*(r1.x-r2.x) + (r1.y-r2.y)*(r1.y-r2.y);
	// 	if (R2 < 1e-10) {
	// 		return 1.0;
	// 	}
	// 	else if (R2 < a*a) {
	// 		return 0.5*R2*log(R2)/a/a;
	// 	}
	// 	else {
	// 		return 0.5*log(R2);
	// 	}
	// }

	// dtype getMatrixEntry(const unsigned i, const unsigned j) {
	// 	pts2D r1 = particles_X[i];//particles_X is a member of base class FMM_Matrix
	// 	pts2D r2 = particles_X[j];//particles_X is a member of base class FMM_Matrix
	// 	double R2	=	(r1.x-r2.x)*(r1.x-r2.x) + (r1.y-r2.y)*(r1.y-r2.y);
	// 	double R	=	sqrt(R2);
  //   // if (i==j) {
  //   //   return 1;//10.0*exp(I*1.0*R);
  //   // }
  //   	// if (R < a) {
  // 		// 	return R/a+1.0;
  // 		// }
  // 		// else {
  //     //   return exp(I*1.0*R)/R;
  // 		// }
  //     return exp(I*1.0*R);
	// }

  ~userkernel() {};
};

// class userkernel: public kernel {
// public:
// 	dtype RHSFunction(const pts2D r) {
// 		dtype q = -1.0+I;
// 		return q;
// 	};
// 	userkernel(std::vector<pts2D>& particles_X, std::vector<pts2D>& particles_Y): kernel(particles_X, particles_Y) {
// 		isTrans		=	true;
// 		isHomog		=	true;
// 		isLogHomog	=	false;
// 		alpha		=	-1.0;
// 	};
// 	// dtype getMatrixEntry(const unsigned i, const unsigned j) {
// 	// 	pts2D r1 = particles_X[i];//particles_X is a member of base class FMM_Matrix
// 	// 	pts2D r2 = particles_X[j];//particles_X is a member of base class FMM_Matrix
// 	// 	double R2	=	(r1.x-r2.x)*(r1.x-r2.x) + (r1.y-r2.y)*(r1.y-r2.y);
// 	// 	double R	=	sqrt(R2);
// 	// 	if (R < a) {
// 	// 		return R/a+1.0;
// 	// 	}
// 	// 	else {
// 	// 		return (1.0+I)*a/R;
// 	// 	}
// 	// }
//
// 	dtype getMatrixEntry(const unsigned i, const unsigned j) {
// 		pts2D r1 = particles_X[i];//particles_X is a member of base class FMM_Matrix
// 		pts2D r2 = particles_X[j];//particles_X is a member of base class FMM_Matrix
// 		double R2	=	(r1.x-r2.x)*(r1.x-r2.x) + (r1.y-r2.y)*(r1.y-r2.y);
// 		double R	=	sqrt(R2);
// 		return exp(I*1.0*R);
// 	}
//
// 	// #elif LOGR
// 	// userkernel(std::vector<pts2D> particles_X, std::vector<pts2D> particles_Y): kernel(particles_X, particles_Y) {
// 	// 	isTrans		=	true;
// 	// 	isHomog		=	false;
// 	// 	isLogHomog	=	true;
// 	// 	alpha		=	1.0;
// 	// };
// 	// double getMatrixEntry(const unsigned i, const unsigned j) {
// 	// 	pts2D r1 = particles_X[i];//particles_X is a member of base class FMM_Matrix
// 	// 	pts2D r2 = particles_X[j];//particles_X is a member of base class FMM_Matrix
// 	// 	double R2	=	(r1.x-r2.x)*(r1.x-r2.x) + (r1.y-r2.y)*(r1.y-r2.y);
// 	// 	if (R2 < 1e-10) {
// 	// 		return 1.0;
// 	// 	}
// 	// 	else if (R2 < a*a) {
// 	// 		return 0.5*R2*log(R2)/a/a;
// 	// 	}
// 	// 	else {
// 	// 		return 0.5*log(R2);
// 	// 	}
// 	// }
// 	// #endif
// 	~userkernel() {};
// };
#endif
