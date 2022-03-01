#include "HODLR_Matrix.hpp"
#include "HODLR.hpp"
#include "KDTree.hpp"
#include "common.hpp"
#include "LS_GMRES.cpp"
#include "LS_HODLR.cpp"
typedef std::complex<double> Dtype;
#include "gmres.hpp"

void writeMatrixToFile(Mat& matrix, std::string filename) {
	//create directory
	std::ofstream myfile;
	myfile.open(filename.c_str());
	myfile << std::setprecision(16);
	// myfile << std::setprecision( std::numeric_limits<int>::max() );
	myfile << matrix << endl;
}

// void ReadFromTextFile(Mat &matrix, std::string filename) {
// 	std::ifstream inFile (filename,std::ios::in);
// 	if(!inFile.good()) {
// 		std::cout<<"Error: could not open file:\""<<filename<<"\"for reading \n";
// 		exit (2);
// 	}
// 	//find the no of values in file
// 	std::istream_iterator<std::string> in{inFile};
// 	std::istream_iterator<std::string> end;
// 	long numberofWords=std::distance(in,end);
// 	//find the no of lines in file
// 	inFile.clear();
// 	inFile.seekg(0,std::ios::beg);
// 	long numberofLines=std::count(std::istreambuf_iterator<char>(inFile),std::istreambuf_iterator<char>(),'\n');
// 	//std::cout<<"no. of words : "<<numberofWords<<"numberofLines: "<<numberofLines<<std::endl;
// 	long rows=numberofLines;
// 	long cols=numberofWords/numberofLines;
// 	if(rows*cols!=numberofWords) {
// 		std::cout<<"\n Infile"<<filename<<"cannot form a matrix \n";
// 		exit(2);
// 	}
// 	matrix.array().resize(rows,cols);
// 	//matrix Base does not allow resizing ...hence change array base
// 	inFile.clear();
// 	inFile.seekg(0,std::ios::beg);
// 	for(unsigned int i=0;i<matrix.rows();i++)
// 	 for(unsigned int j=0;j<matrix.cols();j++)
// 		 inFile>>matrix(i,j);
// 	inFile.close();
// }

// void readMFromFile(Mat &A, std::string filename) {
// 	string Filename = "Validation/A_" + std::to_string(Qchoice) + "_" + std::to_string(N) + "/A";
// 	ReadFromTextFile(A, Filename);
// }

int main(int argc, char* argv[]) {
	int nCones_LFR			=	atoi(argv[1]);
	int nChebNodes			=	atoi(argv[2]);
	int treeAdaptivity	=	atoi(argv[3]);
	double L						=	atof(argv[4]);
	kappa 							= atof(argv[5]);
	int yes2DFMM				=	atoi(argv[6]);
	int degreeOfBases 	= atoi(argv[7]);
	int TOL_POW					= atoi(argv[8]);
	int m 							= atoi(argv[9]);
	int restart					= atoi(argv[10]);
	int nLevelsUniform  = 4;
	double pre_conditioner_tolerance  = atof(argv[11]); // rank or tolerance
	int preconditioner_target_rank = atof(argv[12]);
	Qchoice					    =	atoi(argv[13]); //global variable

	cout << "Wavenumber:		" << kappa << endl;
	cout << "Wavelength:		" << 2*PI/kappa << endl;
	cout << "no. of full cycles:	" << L*kappa/2*PI << endl;
  inputsToDFMM inputs;
  inputs.nCones_LFR = nCones_LFR;
  inputs.nChebNodes = nChebNodes;
  inputs.L = L;
  inputs.yes2DFMM = yes2DFMM;
  inputs.TOL_POW = TOL_POW;
	inputs.degreeOfBases = degreeOfBases;
  inputs.treeAdaptivity = treeAdaptivity;
	inputs.nLevelsUniform = nLevelsUniform;
  Vec Phi;
	double timeIn_getMatrixEntry_Offset = 0.0;
	double timeIn_getMatrixEntry;
	double start, end;

	///////////////////////// GMRES initialisation /////////////////////////////////
	cout << "Begining GMRES initialisation" << endl;
	start		=	omp_get_wtime();
  DFMM *S = new DFMM(inputs);
	S->FMMTreeInitialisation();
	end		=	omp_get_wtime();
	double timeAssemble =	(end-start);
	std::cout << std::endl << "Time taken to assemble: " << timeAssemble << std::endl;
  // defining RHS
  int N = S->K->N;
  Vec b(N);// = Vec::Random(N);//incidence field
	for (size_t i = 0; i < N; i++) {
		b(i) = S->mykernel->RHSFunction(S->gridPoints[i]); //exp(I*kappa*S->gridPoints[i].x);
	}
	cout << "GMRES initialisation done" << endl;
	/////////////////////////////////////////////////////////////////////////



	/////////////////////// HODLR pre-conditioner initialisation /////////////////////
  // double pre_conditioner_tolerance  	= pow(10, -4); //preconditioner tolerance
	std::vector<pts2D> particles_X;//locations
  std::vector<pts2D> particles_Y;//dummy values
	userkernel* mykernel		=	new userkernel();
  H_FMM2DTree<userkernel>* F	=	new H_FMM2DTree<userkernel>(mykernel, nCones_LFR, nChebNodes, L, yes2DFMM, TOL_POW, particles_X, particles_Y, kappa, degreeOfBases, treeAdaptivity, nLevelsUniform);
  int H_N, M, dim;
  H_N = F->gridPoints.size();
  M = F->rank;
  dim = 2;
	Kernel* K = new Kernel(H_N, F);
  bool is_sym = false;
  bool is_pd = false;

	string result_filename;
	std::ofstream myfile0;
	Vec b_tilde;
	double timeGMRES;
	Vec APhi;
	Vec r;
	string filename0;

	// HODLR* T = new HODLR(H_N, M, pre_conditioner_tolerance, preconditioner_target_rank);
  // T->assemble(K, "rookPivoting", is_sym, is_pd);
	// start		=	omp_get_wtime();
  // T->factorize();
	// end		=	omp_get_wtime();
	// double timePreCondFact = end-start;
	// cout << "Time taken to preconditioner factorization: " << timePreCondFact << endl;
	// cout << "HODLR pre-conditioner initialisation done" << endl;
	/////////////////////////////////////////////////////////////////////////
	//mkdir
	result_filename = "result";
	string currPath = std::experimental::filesystem::current_path();
	char final[256];
	sprintf (final, "%s/%s", currPath.c_str(), result_filename.c_str());
	mkdir(final, 0775);

	result_filename = "result/result_" + std::to_string(Qchoice) + "_" + std::to_string(nChebNodes) + "_" + std::to_string(treeAdaptivity) + "_" + std::to_string(int(kappa)) + "_" + std::to_string(m);
	currPath = std::experimental::filesystem::current_path();
	sprintf (final, "%s/%s", currPath.c_str(), result_filename.c_str());
	mkdir(final, 0775);

  // running GMRES
	start		=	omp_get_wtime();
	classGMRES* G = new classGMRES();
	Phi = Vec::Zero(N); // initial
	int maxIterations = 400;
	double gmres_tolerance = 1.0e-12;
	double errGMRES;
	int noOfIterations;
	// G->gmres(S, T, b, maxIterations, gmres_tolerance, Phi, errGMRES, noOfIterations);
	end		=	omp_get_wtime();
	timeGMRES =	(end-start);
	std::cout << "GMRES Residual error(relative): " << errGMRES << std::endl;

	/////////////////////////////////////////////////////////////////
	string MFilename = "Validation";
	char final1[256];
	sprintf (final1, "%s/%s", currPath.c_str(), MFilename.c_str());
	mkdir(final1, 0775);

	MFilename = "Validation/A_" + std::to_string(Qchoice) + "_" + std::to_string(N);
	char final2[256];
	sprintf (final2, "%s/%s", currPath.c_str(), MFilename.c_str());
	mkdir(final2, 0775);

	string Validation_filename = MFilename + "/A";

	// validation of Solver
	//Random Phi: Phi_true
	Vec Validation_True_Phi = Vec::Random(N);
	Vec Validation_True_RHS, Validation_GMRES_Phi, Validation_Hybrid_Phi;
	Mat Validation_True_A(N,N);

	//////////////////////////////////////////////////////////////////
	// for (size_t i = 0; i < N; i++) {
	// 	for (size_t j = 0; j < N; j++) {
	// 		Validation_True_A(i,j) = S->K->getMatrixEntry2(i,j);
	// 	}
	// }
	// writeMatrixToFile(Validation_True_A, Validation_filename);
	//////////////////////////////////////////////////////////////////

	// ReadFromTextFile(Validation_True_A, Validation_filename);
	Validation_True_RHS = Validation_True_A * Validation_True_Phi;

	/////////////// validation of Mat-Vec Product ////////////////////
	Vec trueRHS;
	Vec Validation_DFMM_RHS;
	Mat trueMat;
	start = omp_get_wtime();
	S->MatVecProduct(Validation_True_Phi, Validation_DFMM_RHS);
	end = omp_get_wtime();
	double timeMatVecValidation =	(end-start);
	// S->MatVecProduct(Validation_True_Phi, Validation_DFMM_RHS, trueRHS, trueMat);
	// Vec temp02 = trueRHS-Validation_True_RHS;
	// std::cout << "temp02: " << temp02.norm() << std::endl;
	// Mat temp03 = trueMat-Validation_True_A;
	// std::cout << "temp03: " << temp03.norm() << std::endl;
	Vec temp0 = Validation_DFMM_RHS - Validation_True_RHS;
	// std::cout << "Validation_DFMM_RHS: " << std::endl << Validation_DFMM_RHS << std::endl;
	// std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl << std::endl;
	// std::cout << "Validation_True_RHS: " << std::endl << Validation_True_RHS << std::endl;
	// std::cout << "error: " << std::endl << temp0 << std::endl;
	double Validation_MatVecProduct_Err = temp0.norm()/Validation_True_RHS.norm();
	std::cout << "Validation_MatVecProduct_Err (Relative, Forward): " << Validation_MatVecProduct_Err << std::endl;
	std::cout << "Validation_MatVecProduct_Err (Absolute, Forward): " << temp0.norm() << std::endl << std::endl;
	std::cout << "Time_MatVec_Validation: " << timeMatVecValidation << std::endl;
	// exit(0);
	/////////////////////////////////////////////////////////////////

	// start = omp_get_wtime();
	// G->gmres(S, T, Validation_True_RHS, maxIterations, gmres_tolerance, Validation_Hybrid_Phi, errGMRES, noOfIterations);
	// end = omp_get_wtime();
	// double timeHybridValidation =	(end-start);
	// Vec temp1 = Validation_Hybrid_Phi - Validation_True_Phi;
	// double Validation_Hybrid_Solver_Err = temp1.norm()/Validation_True_Phi.norm();
	// std::cout << "err Hybrid: " << errGMRES << ";	noOfIterations: " << noOfIterations << std::endl;
	// std::cout << "Validation_Hybrid_Solver_Err (Relative, Forward): " << Validation_Hybrid_Solver_Err << std::endl;
	// std::cout << "Validation_Hybrid_Solver_Err (Absolute, Forward): " << temp1.norm() << std::endl;
	// Vec temp2 = Validation_True_RHS - Validation_True_A * Validation_Hybrid_Phi;
	// std::cout << "Validation_Hybrid_Solver_Err (Relative, Backward): " << temp2.norm()/Validation_True_RHS.norm() << std::endl;
	// std::cout << "Validation_Hybrid_Solver_Err (Absolute, Backward): " << temp2.norm() << std::endl;
	// std::cout << "Time_Hybrid_Solver: " << timeHybridValidation << std::endl;
	//
	//
	// start = omp_get_wtime();
	// G->gmres(S, Validation_True_RHS, maxIterations, gmres_tolerance, Validation_GMRES_Phi, errGMRES, noOfIterations);
	// end = omp_get_wtime();
	// double timeGMRESValidation =	(end-start);
	// Vec temp3 = Validation_GMRES_Phi - Validation_True_Phi;
	// double Validation_GMRES_Solver_Err = temp3.norm()/Validation_True_Phi.norm();
	// std::cout << "err GMRES: " << errGMRES << ";	noOfIterations: " << noOfIterations << std::endl;
	// std::cout << "Validation_GMRES_Solver_Err (Relative, Forward): " << Validation_GMRES_Solver_Err << std::endl;
	// std::cout << "Validation_GMRES_Solver_Err (Absolute, Forward): " << temp3.norm() << std::endl;
	// Vec temp4 = Validation_True_RHS - Validation_True_A * Validation_GMRES_Phi;
	// std::cout << "Validation_GMRES_Solver_Err (Relative, Backward): " << temp4.norm()/Validation_True_RHS.norm() << std::endl;
	// std::cout << "Validation_GMRES_Solver_Err (Absolute, Backward): " << temp4.norm() << std::endl;
	// std::cout << "Time_GMRES_Solver: " << timeGMRESValidation << std::endl;

	/////////////////////////////////////////////////////////////////
	/*
	// S->MatVecProduct(Phi, APhi);
	// r = b - APhi;
	// cout << "Err in GMRES: " << r.norm()/b.norm() << endl;
	filename0 = result_filename + "/Phi_1_10_P";
	myfile0.open(filename0.c_str());
	for (size_t l = 0; l < Phi.size(); l++) {
		myfile0 << Phi(l) << endl;
	}
	myfile0.close();
	std::cout << "Time taken by GMRES Solver 1,10: " << timeGMRES << std::endl << endl;

	delete T;
	cout << "------------------------------------------------------" << endl;
	exit(0);
	//////////////////// FIND SCATTERED FIELD FROM PHI ///////////////////////////////////
	Vec U_incidence(N);
	DFMM *S2 = new DFMM(inputs);
	S2->K->findPhi = false;
	S2->FMMTreeInitialisation();
	for (size_t i = 0; i < N; i++) {
		U_incidence(i) = S2->mykernel->IncidenceFunction(S2->gridPoints[i]); //exp(I*kappa*S->gridPoints[i].x);
	}

	Vec U_scattered;
	start		=	omp_get_wtime();
  S2->MatVecProduct(Phi, U_scattered);
	end		=	omp_get_wtime();
	double timeMatVec =	(end-start);
	std::cout << std::endl << "Time taken for MatVec product: " << timeMatVec << std::endl;
	std::cout << std::endl << "Time taken for field computation: " << timeGMRES+timeMatVec << std::endl;

	Vec U_total = U_incidence + U_scattered;
	////////////// write result to file ///////////////////
	string filename;
	filename = result_filename + "/gridPointsX";
	std::ofstream myfile1;
	myfile1.open(filename.c_str());
	for (size_t l = 0; l < S->gridPoints.size(); l++) {
		myfile1 << S->gridPoints[l].x << endl;
	}
	myfile1.close();

	filename = result_filename + "/gridPointsY";
	std::ofstream myfile2;
	myfile2.open(filename.c_str());
	for (size_t l = 0; l < S->gridPoints.size(); l++) {
		myfile2 << S->gridPoints[l].y << endl;
	}
	myfile2	.close();

	filename = result_filename + "/solutionR";
	std::ofstream myfile3;
	myfile3.open(filename.c_str());
	for (size_t l = 0; l < U_total.size(); l++) {
		myfile3 << U_total(l).real() << endl;
	}
	myfile3.close();

	filename = result_filename + "/solutionI";
	std::ofstream myfile4;
	myfile4.open(filename.c_str());
	for (size_t l = 0; l < U_total.size(); l++) {
		myfile4 << U_total(l).imag() << endl;
	}
	myfile4.close();

	filename = result_filename + "/leftRightBoundary"; //to plot
	std::ofstream myfile5;
	myfile5.open(filename.c_str());
	for (size_t l = 0; l < S2->K->leftRightBoundary.size(); l++) {
		myfile5 << S2->K->leftRightBoundary[l].x << "	" << S2->K->leftRightBoundary[l].y << endl;
	}
	myfile5.close();

	filename = result_filename + "/bottomTopBoundary"; //to plot
	std::ofstream myfile6;
	myfile6.open(filename.c_str());
	for (size_t l = 0; l < S2->K->bottomTopBoundary.size(); l++) {
		myfile6 << S2->K->bottomTopBoundary[l].x << "	" << S2->K->bottomTopBoundary[l].y << endl;
	}
	myfile6.close();

	delete S;
	delete S2;
	*/
	////////////////////////////////////////////
}
