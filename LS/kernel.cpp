#include "LS_Assembly.hpp"

class Kernel {
private:
  LS2DTree* F;
  std::vector<pts2D> particles_X;//dummy
  std::vector<pts2D> particles_Y;//dummy
  int nCones_LFR;//number of cones to be used in DAFMM
  double TOL_POW;//currently not in use
  int nLevelsUniform; //number of levels to be constructed in case of uniform tree
  int yes2DFMM; //DFMM or FMM

public:
  int N;
  std::vector<pts2D> gridPoints; // location of particles in the domain
  Kernel(int nChebNodes, int treeAdaptivity, int degreeOfBases, double L) {
    this->nCones_LFR			=	16;//number of cones to be used in DAFMM
    this->TOL_POW         = 9;//currently not in use
    this->nLevelsUniform  = 5; //number of levels to be constructed in case of uniform tree
    this->yes2DFMM				=	0; //DFMM or FMM
    this->F	=	new LS2DTree(nCones_LFR, nChebNodes, L, yes2DFMM, TOL_POW, particles_X, particles_Y, kappa, degreeOfBases, treeAdaptivity, nLevelsUniform);
    gridPoints = F->gridPoints;
    this->N = gridPoints.size(); // locations of particles in the domain
	};

  dtype getMatrixEntry(int i, int j) {
    dtype output;
    output = F->getMatrixEntry(i, j);
    return output;
  }

  ~Kernel() {};
};

int main(int argc, char* argv[])
{
  //inputs to userkernel
  int nChebNodes			=	atoi(argv[1]);//number of gridPoints in leaf box in 1D
	int treeAdaptivity	=	atoi(argv[2]);//tolerance for tree adpativity
  kappa 							= atof(argv[3]);//frequency of operation
  int degreeOfBases 	= atoi(argv[4]); //for discretization of Lippmann-Schwinger equation
  double L						=	atof(argv[5]); //domain size
	Qchoice		    			=	atoi(argv[6]); // contrast function choice
  // cout << "Wavenumber:		" << kappa << endl;
  // cout << "Wavelength:		" << 2*PI/kappa << endl;
  // cout << "no. of full cycles:	" << L*kappa/2*PI << endl;

  Kernel* K = new Kernel(nChebNodes, treeAdaptivity, degreeOfBases, L);
  /*
  take-aways from Kernel object
  K->N : (int); number of locations of particles in the domain
  K->getMatrixEntry(i,j) : (dtype); (i,j)th matrix entry; 0 <= i,j < N
  K->gridPoints : (std::vector<pts2D>) locations of particles in the domain
  */
  //////////////////////////    TEST USAGE   ///////////////////////////////
  int N = K->N;
  std::cout << std::endl << "Problem Size(N): " << N << std::endl << std::endl;
  std::cout << "A(" << int(N/2) << "," << int(N/2) << "): " << K->getMatrixEntry(N/2, N/2) << std::endl << std::endl;
  std::cout << "first 10 locations of particles:" << std::endl;
  for (size_t i = 0; i < 10; i++) {
    std::cout << K->gridPoints[i].x << ", " << K->gridPoints[i].y << std::endl;
  }

  delete K;
}
