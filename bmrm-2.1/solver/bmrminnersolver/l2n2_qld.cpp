/* Copyright (c) 2009, NICTA
 * All rights reserved.
 *
 * The contents of this file are subject to the Mozilla Public License Version
 * 1.1 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 * http://www.mozilla.org/MPL/
 *
 * Software distributed under the License is distributed on an "AS IS" basis,
 * WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
 * for the specific language governing rights and limitations under the
 * License.
 *
 */

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "l2n2_qld.hpp"
#include "configuration.hpp"


/* external fortran function */
typedef long* I_t;
typedef double* D_t;
extern "C" 
{
	extern void ql0001_(I_t M, I_t ME, I_t MMAX, I_t N, I_t NMAX, I_t MNN, 
		    D_t C, D_t D, D_t A,D_t B, D_t XL, D_t XU,
		    D_t X, D_t U, I_t IOUT, I_t IFAIL,I_t IPRINT, D_t WAR,
		    I_t LWAR, I_t IWAR, I_t LIWAR);
}


/*
 * Solve the QP problem
 *
 * argmin_x 0.5 x^T Q x + f x
 * 
 * s.t.     a x <= b
 *          l <=x <= u
 *
 * x, f, a, b, l and u are vectors of 'dim' dimensions
 * Q             is a matrix of 'dim' x 'dim'
 * b             is a scalar
 *
 * all inputs are in dense format
 *
 * the output is stored in 'x'
 *
 * solver used: qld
 *
 * note: in the fortran solver used (qld) the constraint
 * is Ax + B >= 0, so we have to invert the sign of 'a'
 */
void CL2N2_qld::SolveQP()
{
	double B[1] = {1};
	double *A;

	A = new double[dim];
	for (int i=0; i<dim; i++)
		A[i]   = -a[i];
	long M    = 1;
	long ME   = 0;
	long MMAX = M;
	long N    = dim;
	long NMAX = dim;
	long MNN  = M+N+N;
	long IOUT = 6;
	long IFAIL = 0;
	long IPRINT = 0;
	long LWAR   =3*NMAX*NMAX/2 + 10*NMAX + 2*MMAX+2; 
	long LIWAR  = N;

  	double *WAR  = new double[LWAR];
	long   *IWAR = new long[LIWAR];
	double *U    = new double[dim*3];

	IWAR[0]=1;

	ql0001_(&M, &ME, &MMAX, &N, &NMAX, &MNN, Q, f, A, B, l, u, x, U, &IOUT, &IFAIL, &IPRINT, WAR, &LWAR, IWAR, &LIWAR);

	if (IFAIL!=0) printf("IFAIL = %ld!\n", IFAIL);

	delete[] A;
	delete[] WAR;
	delete[] IWAR;
	delete[] U;
}
