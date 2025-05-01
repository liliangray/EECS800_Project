//Functions for LU solver

//Solve a system using backward substitution (serial)
vector<double> backSubSolve(
  const vector<vector<double>>& upperMatA,
  const vector<double>& vecB
){
	int sizeA = upperMatA.size();
	double tol = 1e-12;
	if(vecB.size() != sizeA){
		cerr << "backSubSolve(): Sizes incompatible for solving via backward substitution.\n\n";
		exit(1);
	}
	vector<double> result(sizeA, 0.0);
	for(int i = sizeA - 1; i >= 0; i--){
		double sum = 0.0;
		for(int j = i + 1; j < sizeA; j++){
			sum += upperMatA[i][j] * result[j];
		}
		if(fabs(upperMatA[i][i]) < tol){
			cerr << "backSubSolve(): Matrix is singular at element (" << i << ", "<< i << ").\n\n";
			exit(1);
		}
		result[i] = (vecB[i] - sum) / upperMatA[i][i];
	}
	return result;
}

//Solve a system using forward substitution (serial)
vector<double> forwardSubSolve(
  const vector<vector<double>>& lowerMatA,
  const vector<double>& vecB
){
	int sizeA = lowerMatA.size();
	double tol = 1e-12;
	if (vecB.size() != sizeA) {
		cerr << "forwardSubSolve(): Sizes incompatible for solving via forward substitution.\n\n";
		exit(1);
	}
	vector<double> result(sizeA, 0.0);   
	for(int i = 0; i < sizeA; i++){
		double sum = 0.0;
		for(int j = 0; j < i; j++){
			sum += lowerMatA[i][j] * result[j];
		}
		result[i] = (vecB[i] - sum)/lowerMatA[i][i];
	}
	return result;
}

//LU Solver
vector<double> solverLU(
  vector<vector<double>>& A, // n by n matrix
  vector<double>& b // vector of length n
){
	size_t n = A.size();
	double tol = 1e-12;
	vector<vector<double>> U(n, vector<double>(n, 0.0));
	vector<vector<double>> L(n, vector<double>(n, 0.0));
	vector<double> m(n, 0.0); //Temporary vector for multipliers
	//Pivoting
	size_t p = 0;
	size_t q = 0;
	while((p < n) && (q < n)){ //find pivot
		size_t iMax = p;
		double AcolMax = fabs(A[iMax][q]);
		for(size_t i = p + 1; i < n; i++){
			double temp = fabs(A[i][q]);
			if(temp > AcolMax){
				AcolMax = temp;
				iMax = i;
			}
		}
		if(A[iMax][q] == 0.0){
			q++;
		}else{ //swap rows
			swap(A[p], A[iMax]);
			swap(b[p], b[iMax]);
			p++;
			q++;
		}
	}
	//Gaussian Elimination
	for(size_t k = 0; k < n - 1; k++){
		//Compute multipliers
		for(size_t i = k + 1; i < n; i++){
			if(fabs(A[k][k]) < tol){
				cerr << "solverLU(): Matrix is singular/nerly singular, cannot perform LU decomposition.\n\n";
				exit(1);
			}
			m[i] = A[i][k] / A[k][k];
		}
		//Apply transformation to rest of matrix
		for(size_t j = k + 1; j < n; j++){
			for(size_t i = k + 1; i < n; i++){
				A[i][j] -= m[i] * A[k][j];
			}
		}
	}
	//Fill U
	for(size_t i = 0; i < n; i++){
		for(size_t j = i; j < n; j++){
			U[i][j] = A[i][j];
		}
	}
	//Fill L
	for(size_t i = 0; i < n; i++){
		for(size_t j = 0; j < i; j++){
			L[i][j] = A[i][j];
		}
		L[i][i] = 1.0;
	}
	vector<double> y = forwardSubSolve(L, b);
	vector<double> x = backSubSolve(U, y);
	return x;
}