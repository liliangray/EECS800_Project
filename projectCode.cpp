/* Functions in this file: 
 *
 * dotProd - parallel 
 * matMult - parallel
 * matVecMult - parallel 
 * serDotProd - serial (not sure if we need this one)
 * serMatVecMult - serial (not sure if we need this one)
 * vecMult - parallel 
 * vecMatMult - parallel 
 * phi - serial 
 * phiPrime - serial 
 * lineL - serial 
 * ArmijoCondi - serial 
 * curvCondi - serial 
 * WolfeCondi - serial 
 * GSLS - serial 
 * parBFGS - parallel-sh (uses parallel functions but not explicitly parallel yet)
 * 
 * Need: global dimension of problem? Also functionToMin() and gradF() functions for GSLS computations 
 * 
 */

#include <mpi.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <iomanip>
#include <stdio.h>
#include <cstdlib>
#include <string> 
#include <set> 
#include <fstream>
#include <map>
#include <sstream>
#include <functional>
#include <ctime>

using namespace std;

//Calculate dot product 
//(parallel)
double dotProd(const vector<double>& v1, const vector<double>& v2){ 
	int pid, np;
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	const int ROOT = 0;
	
	double localDot = 0.0;
	double totalDot = 0.0;
	int errorFlag = 0;
	size_t n = v1.size();
	if(pid == ROOT){
		if(n != v2.size()){
			cerr << "dotProd(): Vector sizes incompatible for dot product.\n\n";
			errorFlag = 1;
		}
	}
	MPI_Bcast(&errorFlag, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
	if(errorFlag != 0){
		MPI_Abort(MPI_COMM_WORLD, 1);
		return NAN;
	}
	MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, ROOT, MPI_COMM_WORLD);
	size_t nPerNP = n/np; //note that if np > n, some procs will be idle
	int extra = n%np;
	int startRow = (pid * nPerNP) + min(pid, extra);
	size_t endRow = startRow + nPerNP + (pid < extra ? 1 : 0);
	if(startRow < n){
		for(size_t i = startRow; i < endRow; i++){
			localDot += v1[i]*v2[i];
		}
	}
	MPI_Allreduce(&localDot, &totalDot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	return totalDot;
}

//Compute matrix-matrix multiplication 
//(parallel)
vector<vector<double>> matMult(
  const vector<vector<double>>& a,
  const vector<vector<double>>& b
){
	int pid, np;
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	const int ROOT = 0;
	
	size_t arows = a.size();
	size_t acols = a[0].size();
	size_t brows = b.size();
	size_t bcols = b[0].size();
	size_t r = arows;
	size_t c = bcols;
	size_t rPerNP = r/np; //note that if np > r, some procs will be idle
	int extra = r%np;
	int startRow = (pid * rPerNP) + min(pid, extra);
	size_t endRow = startRow + rPerNP + (pid < extra ? 1 : 0);
	size_t localRows = endRow - startRow;
	//Compute local results
	vector<vector<double>> localResult(localRows, vector<double>(c, 0.0));
	if(startRow < r){
		for(size_t i = 0; i < localRows; i++){
			for(size_t j = 0; j < c; j++){
				for(size_t k = 0; k < acols; k++){
					localResult[i][j] += a[startRow + i][k] * b[k][j];
				}
			}
		}
	}
	vector<double> flatLocalResult(localRows * c, 0.0);
	//"Flatten" localResult for MPI_Gatherv
	for(size_t i = 0; i < localRows; i++){
		for(size_t j = 0; j < c; j++){
			flatLocalResult[i * c + j] = localResult[i][j];
		}
	}
	//Gather results from all processes
	vector<int> recvCounts(np, rPerNP * c);
	for(size_t i = 0; i < extra; i++){
		recvCounts[i] += c;
	}
	vector<int> displs(np, 0);
	for(size_t i = 1; i < np; i++){
		displs[i] = displs[i - 1] + recvCounts[i - 1];
	}
	vector<double> flatResult(r * c, 0.0);
	MPI_Gatherv(flatLocalResult.data(), flatLocalResult.size(), MPI_DOUBLE, flatResult.data(), recvCounts.data(), displs.data(), MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	//Broadcast the result matrix to all processes and reconstruct result from flat array 
	MPI_Bcast(flatResult.data(), r * c, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	vector<vector<double>> result(r, vector<double>(c, 0.0));
	for(size_t i = 0; i < r; i++){
		for(size_t j = 0; j < c; j++){
			result[i][j] = flatResult[i * c + j];
		}
	}
	return result;
}

//Compute matrix-vector multiplication 
//(parallel)
vector<double> matVecMult(
  const vector<vector<double>>& m, 
  const vector<double>& v
){
	int pid, np;
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	const int ROOT = 0;
	
	size_t rows = m.size();
	size_t cols = m[0].size();
	size_t n = v.size();
	int errorFlag = 0;
	if(pid == ROOT){
		if(n != cols){
			cerr << "matVecMult(): Sizes incompatible for multiplication.\n\n";
			errorFlag = 1;
		}
	}
	MPI_Bcast(&errorFlag, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
	if(errorFlag != 0){
		MPI_Finalize(); 
		exit(1);
	}

	size_t rPerNP = rows/np; //note that if np > rows, some procs will be idle
	int extra = rows%np;
	int startRow = pid * rPerNP + min(pid, extra);
	size_t endRow = startRow + rPerNP + (pid < extra ? 1 : 0);
    //Compute local results
	vector<double> localResult((endRow - startRow), 0.0);
	if(startRow < rows){
		for(size_t i = startRow; i < endRow; i++){
			for(size_t j = 0; j < cols; j++){
				localResult[i - startRow] += m[i][j] * v[j];
			}
		}
	}
	//Gather results from all processes
	vector<double> result(rows, 0.0);
	vector<int> recvCounts(np, rPerNP);
	for(size_t i = 0; i < extra; i++){
		recvCounts[i]++;
	}
	vector<int> displs(np, 0);
	for(int i = 1; i < np; ++i){
		displs[i] = displs[i - 1] + recvCounts[i - 1];
	}
	MPI_Gatherv(localResult.data(), localResult.size(), MPI_DOUBLE, result.data(), recvCounts.data(), displs.data(), MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	//Broadcast the result to all processes
	MPI_Bcast(result.data(), result.size(), MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	return result;
}

//Calculate dot product 
//(serial)
double serDotProd(const vector<double>& v1, const vector<double>& v2){ 
	if(v1.size() != v2.size()){
		cerr << "serDotProd(): Vectors must be the same size.\n\n";
		exit(1);
	}
	double result = 0.0;
	for(size_t i = 0; i < v1.size(); i++){
		result += v1[i]*v2[i];
	}
	return result;
}

//Compute matrix-vector multiplication 
//(serial)
vector<double> serMatVecMult(const vector<vector<double>>& m, const vector<double>& v){
	size_t mRows = m.size();
	size_t mCols = m[0].size();
	vector<double> result(mRows, 0.0);
	for(size_t i = 0; i < mRows; i++){
		for(size_t j = 0; j < mCols; j++){
			result[i] += m[i][j]*v[j];
		}
	}
	return result;
}

//Compute vector-vector multiplication that results in a matrix 
//(parallel)
vector<vector<double>> vecMult(
  const vector<double>& v1,
  const vector<double>& v2
){
    int pid, np;
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	const int ROOT = 0;
	
	size_t rows = v1.size();
	size_t cols = v2.size();
	size_t total = rows * cols;
	size_t rPerNP = rows/np; //note that if np > rows, some procs will be idle
	int extra = rows%np;
	int startRow = pid * rPerNP + min(pid, extra);
	size_t endRow = startRow + rPerNP + (pid < extra ? 1 : 0);
	//Compute local results
    vector<vector<double>> localResult(endRow - startRow, vector<double>(cols, 0.0));
	//if(startRow < rows){
		for(size_t i = startRow; i < endRow; i++){
			for(size_t j = 0; j < cols; j++){
				localResult[i - startRow][j] = v1[i] * v2[j];
			}
		}
	//}
	//Gather results from all processes
	vector<int> recvCounts(np, rPerNP * cols);
	for(size_t i = 0; i < extra; i++){
		recvCounts[i] += cols;
	}
	vector<int> displs(np, 0);
	for(int i = 1; i < np; i++){
		displs[i] = displs[i - 1] + recvCounts[i - 1];
	}
	vector<double> flatResult(total, 0.0);
	vector<double> flatLocalResult((endRow - startRow) * cols, 0.0);
	//Flatten localResult for MPI_Gatherv
	for(size_t i = 0; i < localResult.size(); i++){
		for(size_t j = 0; j < localResult[i].size(); j++){
			flatLocalResult[i * cols + j] = localResult[i][j];
		}
	}
	MPI_Gatherv(flatLocalResult.data(), flatLocalResult.size(), MPI_DOUBLE, flatResult.data(), recvCounts.data(), displs.data(), MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	//Reconstruct the result matrix from the flat result array on all processes
	MPI_Bcast(flatResult.data(), total, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	vector<vector<double>> result(rows, vector<double>(cols));
	for(size_t i = 0; i < rows; i++){
		for(size_t j = 0; j < cols; j++){
			result[i][j] = flatResult[i * cols + j];
		}
	}
	return result;
}

//Compute v^T * M = vector of size 1 x m 
//(parallel)
vector<double> vecMatMult(
  const vector<double>& v, 
  const vector<vector<double>>& M
){
	int pid, np;
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	const int ROOT = 0;

	size_t n = v.size(); //number of rows in M, and entries in v
	size_t m = M[0].size(); //number of columns in M

	size_t rowsPerProc = n / np;
	int extra = n % np;
	size_t startRow = pid * rowsPerProc + std::min(pid, extra);
	size_t endRow = startRow + rowsPerProc + (pid < extra ? 1 : 0);
	size_t localRows = endRow - startRow;

	//Local result: accumulate partial sums for each column
	vector<double> localResult(m, 0.0);
	for (size_t i = startRow; i < endRow; ++i) {
		for (size_t j = 0; j < m; ++j) {
			localResult[j] += v[i] * M[i][j];
		}
	}

	//Global result (on ROOT)
	vector<double> globalResult(m, 0.0);
	MPI_Reduce(localResult.data(), globalResult.data(), m, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

	//Broadcast result to all processes
	MPI_Bcast(globalResult.data(), m, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

	return globalResult; //1 x m row vector
}



////////////////////////////////////// GSLS functions 

//Line search function: phi(alpha) = f(x_k + alpha*d) 
//(serial)
double phi( 
  vector<double> x_k, //current values of function inputs
  vector<double> d, //line search direction
  double alpha //a positive scalar
){
	vector<double> xVal(dim, 0.0); //vector of f inputs (x_k + alpha*d)
	for(int i = 0; i < dim; i++){
		xVal[i] = x_k[i] + alpha*d[i]; 
	}
	double phi = functionToMin(xVal);////////////////////
	return phi;
}

//Derivative of the line search function: phi'(alpha) = grad(f(x_k + alpha*d)) * d
//(serial)
double phiPrime( 
  vector<double> x_k, //current values of function inputs
  vector<double> d, //line search direction
  double alpha //a positive scalar
){
	vector<double> xVal(dim, 0.0);
	for(int i = 0; i < dim; i++){
		xVal[i] = x_k[i] + alpha*d[i];
	}
	vector<double> partials(dim, 0.0); 
	partials = gradF(xVal);////////////////////
	double phiPrime = serDotProd(partials, d);
	return phiPrime;
}

//A straight line with slope less negative than phi'(0): L(alpha) = phi(0) + (c1 * phi'(0) * alpha)
//(serial)
double lineL(
  vector<double> x_k, //current values of function inputs
  vector<double> d, //line search direction
  double alpha, //a positive scalar
  double c1 //constant 1 for Wolfe Conditions
){
	double phi0 = phi(x_k, d, 0.0);
	double phiPrime0 = phiPrime(x_k, d, 0.0);
	return phi0 + (c1 * phiPrime0 * alpha);
}

//Sufficient Decrease: the Armijo Condition (Wolfe Condition #1)
//(serial)
bool ArmijoCondi(
  vector<double> x_k, //current values of function inputs
  vector<double> d, //search direction
  double alpha, //a positive scalar
  double c1 //constant 1 for Wolfe Conditions
){
	double phiAlpha = phi(x_k, d, alpha);
	double lineAlpha = lineL(x_k, d, alpha, c1);
	if(phiAlpha <= lineAlpha){
		return true;
	}else{
		return false;
	}
}

//Sufficient Progress: the Curvature Condition (Wolfe Condition #2)
//(serial)
bool curvCondi(
  vector<double> x_k, //current values of function inputs
  vector<double> d, //search direction
  double alpha, //a positive scalar
  double c2, //constant 2 for Wolfe Conditions
  int strongFlag //flag for Strong Wolfe Conditions: 0 = regular/weak, 1 = strong 
){
	double phiPrime0 = phiPrime(x_k, d, 0.0);
	double phiPrimeK = phiPrime(x_k, d, alpha);
	if(strongFlag == 0){ //Normal/Weak
		double check = c2 * phiPrime0;
		if(phiPrimeK >= check){
			return true;
		}else{
			return false;
		}
	}else{ //Strong
		double check = c2 * fabs(phiPrime0);
		if(fabs(phiPrimeK) <= check){
			return true;
		}else{
			return false;
		}
	}
}

//Wolfe Conditions
//(serial)
bool WolfeCondi(
  vector<double> x_k, //current values of function inputs
  vector<double> d, //search direction
  double alpha, //a positive scalar
  double c1, //constant 1 for Wolfe Conditions, 0.0 < c1 < c2
  double c2, //constant 2 for Wolfe Conditions, c1 < c2 < 1.0
  int strongFlag //flag for Strong Wolfe Conditions: 0 = regular/weak, 1 = strong 
){
	bool armijo = ArmijoCondi(x_k, d, alpha, c1); //Armijo Condition
	bool curvature = curvCondi(x_k, d, alpha, c2, strongFlag); //Curvature Condition
	if(armijo && curvature){ //If both are true, Wolfe Conditions are satisfied
		return true;
	}else{ //If one or both is/are false, Wolfe Conditions fail
		return false;
	}
}

//Golden Section Line Search (GSLS)
//(serial)
double GSLS(
  vector<double> x, //starting "point"
  vector<double> d, //search direction
  double aMax, //maximum value of alpha (a positive scalar)
  double c1, //constant 1 for Wolfe Conditions, 0.0 < c1 < c2
  double c2, //constant 2 for Wolfe Conditions, c1 < c2 < 1.0
  int strongFlag, //flag for Strong Wolfe Conditions: 0 = regular/weak, 1 = strong 
  int &lowLevelIter //counter for total number of low-level iterations
){
	double tol = 1e-12; //1e-4
	double min = 0.0;
	bool wolfe = 0;
	const double tau = ((sqrt(5.0)) - 1.0)/2.0;
	double a = 0.0; //right endpoint of uncertainty interval
	double b = aMax; //left endpoint of uncertainty interval
	int j = 1; //iteration counter
	double alpha_j = 0.0; 
	double p1 = a + ((1.0 - tau) * (b - a));
	double p2 = a + (tau * (b - a));
	double phiF1 = phi(x, d, p1);
	double phiF2 = phi(x, d, p2);
	while(!wolfe && (b - a) > tol){
		if(phiF1 > phiF2){
			a = p1;
			p1 = p2;
			phiF1 = phiF2;
			p2 = a + (tau * (b - a));
			phiF2 = phi(x, d, p2);
		}else if (phiF1 < phiF2){
			b = p2;
			p2 = p1;
			phiF2 = phiF1;
			p1 = a + ((1.0 - tau) * (b - a));
			phiF1 = phi(x, d, p1);
		}
		alpha_j = (b + a)/2.0;
		wolfe = WolfeCondi(x, d, alpha_j, c1, c2, strongFlag);
		min = alpha_j;
		if(wolfe){ //Stop if Wolfe Conditions are satisfied
			break;
		}
		j++;
		lowLevelIter++;
	}
	return min;
}


////////////////////////////////////// BFGS  

//Initial attempt at BFGS (4/30/25)
//(parallel-ish)
void parBFGS( //what should the output type be?
  const int dim, //dimension of problem (not needed here if we have a global dim variable)
  vector<double> x_k, //initial guess vector, dim x 1
  vector<vector<double>> B_k //initial Hessian approximation, dim x dim (often the identity matrix)
){
	int pid, np;
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	const int ROOT = 0;
	
	//Initialization of parameters
	vector<double> s_k(dim, 0.0); //quasi-Newton step direction (dim x 1)
	double alpha_k = 0.0; //step size (scalar) 
	vector<double> y_k(dim, 0.0);
	vector<vector<double>> B_new(dim, vector<double>(dim, 0.0)); //B_{k+1}
	vector<double> x_new(dim, 0.0); //x_{k+1}
	//gradf(x_k)
	//gradf(x_{k+1})
	
	//Used in step 4 of for-loop 
	double ykTsk = 0.0; //y_k^T * s_k
	vector<vector<double>> ykykT(dim, vector<double>(dim, 0.0)); //y_k * y_k^T
	vector<double> Bksk(dim, 0.0); //B_k * s_k
	vector<vector<double>> skTBk(1, vector<double>(dim, 0.0)); //s_k^T * B_k
	vector<vector<double>> BkskskTBk(dim, vector<double>(dim, 0.0)); //(B_k * s_k) * (s_k^T * B_k)
	double skTBksk = 0.0; //s_k^T * (B_k * s_k)
	
	for(int k = 0; k < dim; k++){ //check if dim is the correct stopping iter. I don't think it is 
		//1: Solve B_k * s_k = -grad(f(x_k)) for s_k (LU? HHQR?)
		
		
		//2: Use GSLS to determine x_{k+1} = x_k + alpha_k * s_k
		
		
		//3: Compute y_k = grad(f(x_{k+1})) - grad(f(x_k))
		
		
		//4: Compute B_{k+1} = B_k - (B_k * s_k * s_k^T * B_k)/(s_k^T * B_k * s_k) + (y_k * y_k^T)/(y_k^T * s_k)
		ykTsk = dotProd(y_k, s_k); //Compute y_k^T*s_k
		ykykT = vecMult(y_k, y_k); //Compute y_k*y_k^T
		for(int i = 0; i < dim; i++){
			for(int j = 0; j < dim; j++){
				ykykT[i][j] = 1.0/ykTsk * ykykT[i][j]; //Compute (y_k*y_k^T)/(y_k^T*s_k)
			}
		}
		Bksk = matVecMult(B_k, s_k); //Compute B_k*s_k
		skTBk = vecMatMult(s_k, B_k); //Compute s_k^T*B_k
		skTBksk = dotProd(s_k, Bksk); //Compute s_k^T*(B_k*s_k)
		BkskskTBk = vecMult(Bksk, skTBk); //Compute (B_k*s_k)*(s_k^T*B_k)
		for(int i = 0; i < dim; i++){
			for(int j = 0; j < dim; j++){
				BkskskTBk[i][j] = 1.0/skTBksk * BkskskTBk[i][j]; //Compute ((B_k*s_k)*(s_k^T*B_k))/(s_k^T*(B_k*s_k))
			}
		}
		B_new = B_k + ykykT - BkskskTBk; //Compute B_new 
	}
}