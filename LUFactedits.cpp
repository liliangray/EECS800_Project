#include <iostream>
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include "mpi.h"

using namespace std;

void lu(double** A, int n, double** L, double** U, int myrank, int numprocs); 
//would need to input n and specify procs when running MPI

int main( int argc, char* argv[]){
	int n = atoi(argv[1]);
  char* outlfile = argv[2];
  char* outufile = argv[3];
  int myrank, numprocs, i,j;
  double t1,t2;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank( MPI_COMM_WORLD, &myrank );
  MPI_Comm_size( MPI_COMM_WORLD, &numprocs );
  t1 = MPI_Wtime();
  MPI_Status status;
  int errors=0;
  if (numprocs > n) {
    if (myrank == 0) {
      fprintf(stderr,"Number of processors cannot exceed number of matrix rows.");
    }
  MPI_Abort(MPI_COMM_WORLD, 1);
  }
  // Obtain names of output files.
  ofstream outl (outlfile, ios::out);
  ofstream outu (outufile, ios::out);

  // Open output files for writing.
  if (! outl.is_open())
    cout << "Couldn't open output file for writing L." << endl;
    
  if (! outu.is_open())
    cout << "Couldn't open output file for writing U." << endl;
  //create a matrix of size n and partition it
  // Create n-by-n matrix A.
  double** A = (double**) malloc(n*sizeof(double*));
  for ( i=0; i < n; ++i){
    A[i] = (double*) malloc(n*sizeof(double));
  }
       
  // Initialize A to random dense matrix w/ entries between 1 and 10
  for ( i = 0; i < n; ++i){
    for (j = 0; j < n; ++j){
      A[i][j] = ((double) rand() / RAND_MAX) * 9 + 1;
    }
  }
  if(myrank == 0){
    for ( i=0; i < n; ++i){
      for ( j=0; j < n; ++j){
        cout << A[i][j] << " ";
        if (j == (n-1)) {
          cout << endl;
        }
      }
    }
  }
  // Create L matrix.
  double** L = (double**) malloc(n*sizeof(double*));
  for (i=0; i < n; ++i){
    L[i] = (double*) malloc(n*sizeof(double));
  }
  for (i = 0; i < n; ++i){
    for (j = 0; j < n; ++j){
      L[i][j] = 0.0;
    }
  }
  // Create U matrix.
  double** U = (double**) malloc(n*sizeof(double*));
  for (i=0; i < n; ++i){
    U[i] = (double*) malloc(n*sizeof(double));
  }
  for (i = 0; i < n; ++i){
    for (j = 0; j < n; ++j){
      U[i][j] = 0.0;
    }
  }
  //successfully creates A, an nxn dense matrix
  // Compute LU factorization of A.
  lu(A,n,L,U,myrank,numprocs);
  int block = n/numprocs;
  double* L_1d = nullptr;
  double* U_1dim = nullptr;
  double* luprod = nullptr;
  double* L_temp = new double[n*block];
  double* lu_temp = new double[n*block];
  // Write LU factorization to file.
  //wrap output so only printing on p0
  if(myrank == 0){
    outl << setprecision(16) << n << endl;
      for (i=0; i < n; ++i){
        for (j=0; j < n; ++j){
          outl << setprecision(16) << L[i][j] << " ";
          if (j == (n-1)) {
            outl << endl;
          }
        }
      }

    outu << setprecision(16) << n << endl;
      for (i=0; i < n; ++i){
        for (j=0; j < n; ++j){
          outu << setprecision(16) << U[i][j] << " ";
          if (j == (n-1)) {
            outu << endl;
          }
        }
      }

    //compute test for ||A-LU||_2 < tol
    
    double* L_1d = new double[n*n];
    double* U_1dim = new double[n*n];
    //temporary variables for computation
    
    //flatten L and U for computations
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        //flatten Atemp and A
        L_1d[i*n +j] = L[i][j];
        U_1dim[i*n+j] = U[i][j];
      }
    }
  //Send flattened matrices for LU computation
  MPI_Scatter(L_1d, block*n, MPI_DOUBLE, L_temp, block*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(U_1dim, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  double norm_comp = 0.0;
  cout << setprecision(16) << n << endl;
  for (i=0; i<block; i++){
    for (j=0; j<n;j++){
      double lu_value = 0.0;
      for (int k = 0; k<n; k++){
        lu_value += L_temp[i*n+k]*U_1dim[k*n+j];
        //compute a component of the norm
      }
      lu_temp[i*n+j] = lu_value;
      //Compute the sum of squared residuals
      norm_comp += pow(A[i][j]-lu_temp[i*n+j],2); 
    }
  }
  double fnorm;
  //Compute the square root of that sum
  fnorm = sqrt(norm_comp);
  if (myrank==0){
    cout << "The residual error is: " << setprecision(16) << fnorm << endl;
  }
  // Close files.
  outl.close();
  outu.close();
  // Free variables.
  delete[] L_1d; 
  delete[] U_1dim;
  delete[] L_temp;
  delete[] lu_temp;
  } 

  // Free A, L, U.
  for (i=0; i < n; ++i){
    free(A[i]);
    free(L[i]);
    free(U[i]);
  }
  free(A);
  free(L);
  free(U);

  t2 = MPI_Wtime();
  if (myrank == 0){
  printf("The total time elapsed for problem size %d and %d processors was %1.2f\n", n, numprocs, t2-t1);fflush(stdout);
  }
  MPI_Finalize();
  return(EXIT_SUCCESS);
}
void lu(double** A, int n, double** L, double** U, int myrank, int numprocs){
  //make Atemp
  //size of the partitions
  //we will assume this divides nicely
  int blocksize = n/numprocs;
  //compute local rows for each block
  int start_row = myrank*blocksize;
  int end_row = start_row+blocksize;
  //create temp A matrix
  //could be parallelized?
  //double** Atemp = (double**) malloc(n*sizeof(double*));
  //for (int i=0; i < n; ++i){
  //  Atemp[i] = (double*) malloc(n*sizeof(double));
  //}
  double** M = (double**) malloc(n*sizeof(double*));
  for (int i=0; i < n; i++){
    M[i] = (double*) malloc(n*sizeof(double));
  }
  
  
  double* A_1d = new double[n*n];
  //local arrays for storing scattered rows
  double* A_local = new double[blocksize * n]; 
  double* M_local = new double[blocksize*n]();
  double* M_1d = new double[n*n]();
  double* U_1d = new double[n*n]();
  for (int i = 0; i < n*n; i++) {
    M_1d[i] = 0.0;
    U_1d[i] = 0.0;
  }
  
  //flatten matrices for MPI use
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      //Copy A into A_1d
      A_1d[i*n + j] = A[i][j];
    }
  }
  
  //Master processor sends rows of A_1d to worker procs
  MPI_Scatter(A_1d, blocksize*n, MPI_DOUBLE, A_local, blocksize*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  double* pivot = new double[n];
  for (int k = 0; k < n; k++){
    //compute current owner of the block for broadcasting
    int owner = k/blocksize;
    //compute local index in flattened array
    int localindex = k % blocksize;
    //only compute the multiplier for the proc with the pivot row
    if (owner == myrank){
      //printf("Pivot row %d from rank %d:\n", k, myrank);fflush(stdout);
      for (int j = 0; j < n; j++) {
        pivot[j] = A_local[localindex*n+j];
      }
    }
    //broadcast pivot row from the owner
    MPI_Bcast(pivot, n, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    //check for pivot divide by zero
    if(pivot[k]==0){    
      if(myrank == 0){
         printf("divide by zero error");fflush(stdout);
      }
      MPI_Abort(MPI_COMM_WORLD,-1);
    }
    //ij part of loop
    for (int i=start_row; i<end_row; i++){
      //don't worry about lower entries than diagonal for computing the multiplier
      if(i>k){
        double multiplier = A_local[(i-start_row)*n + k] / pivot[k];
        M_local[(i-start_row)*n + k] = multiplier;
        for (int j = k+1; j < n; j++) {
          A_local[(i-start_row)*n+j] -= multiplier*pivot[j];
        }
        
      }
    }
    
  }
  delete[] pivot; 
  cout << "LU factorization finished on rank " << myrank << endl;
  
  // Create U. with Gather
  MPI_Gather(A_local,blocksize*n,MPI_DOUBLE,U_1d, blocksize*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
  //reconstruct M with Gather
  MPI_Gather(M_local, blocksize * n, MPI_DOUBLE, M_1d, blocksize * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //reconstruct U and L
  if(myrank == 0){  
    for (int i=0; i < n; ++i){
      for (int j=0; j < n; ++j){
        if(j >= i){
          U[i][j] = U_1d[i*n+j];
        }
        else{
          L[i][j] = M_1d[i*n+j];
          //This looks good!!
        }
      }
      // Create strictly lower triangular part of L.
      fprintf(stderr,"Starting to write out U\n");fflush(stderr);
      L[i][i] = 1.0;
    }
    //prints U
    fprintf(stderr, "U matrix:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fprintf(stderr, "%f ", U[i][j]);
        }
        fprintf(stderr, "\n");
    }
    //prints L
    fprintf(stderr, "L matrix:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fprintf(stderr, "%f ", L[i][j]);
        }
        fprintf(stderr, "\n");
    }
     
  }
  delete[] U_1d; 
  delete[] M_1d;
  delete[] A_local;
  delete[] A_1d;
  delete[] M_local;
  //for (int i=0; i < n; ++i){
  //  free(Atemp[i]);
  //}
  //free(Atemp);

  // Free M.
  for (int i=0; i < n; ++i){
    free(M[i]);
  }
  free(M);

}






