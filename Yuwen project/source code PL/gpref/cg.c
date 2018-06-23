// conjugate gradient
// min 1/2 x' * A * x + B' * x
// 
#include <stdlib.h> 
#include <malloc.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

int conjugate_gradient (double * A, double * B, double * x, unsigned int n)
{
	double * r ;
	double * ap ;
	double * p ;
	unsigned int loop, i ;
	long unsigned index = 1 ;
	double pap, qij, nr=0, r1=0, r0=0, lambda, beta ; 

	if (NULL ==A || NULL == B || NULL == x || n <= 0)
	{
		printf("input pointers are NULL.\n") ;
		return 1 ;
	}
	if (n>50000)
	{
		printf("input dimension %u is too high.\n", n) ;
		return 1 ;
	}
	
	r = (double *) malloc((n)*sizeof(double)) ;
	ap = (double *) malloc((n)*sizeof(double)) ;
	p = (double *) malloc((n)*sizeof(double)) ;
	
	
	if ( NULL == r || NULL == ap || NULL == p )
	{
		printf("out of memory.\n") ;
		return 1 ;		
	}

	// initialize r, kin, p
	for (loop=0;loop<n;loop++)
	{
		//printf("%f\n",B[loop]) ;
		r[loop] = B[loop] ;
		nr += fabs( r[loop] ) ;
		r1 += r[loop]*r[loop] ; // r0
	}
	nr = nr / (double)(n) ;

	while ( nr > 0.000001 || index == 1 )//
	{
		if (index == 1)
		{
			for (loop=0;loop<n;loop++)
			{
				p[loop] = r[loop] ;
				x[loop] = 0 ;
			}
		}
		else
		{
			beta = r1 / r0 ;
			for (loop=0;loop<n;loop++)
				p[loop] = r[loop] + beta * p[loop] ;
		}
		//Ap
		pap = 0 ;
		for (loop=0;loop<n;loop++)
			ap[loop] = 0 ;

		for (loop=0;loop<n;loop++)
		{
			for (i=loop+1;i<n;i++)
			{
				assert(A[loop*n+i]==A[i*n+loop]) ;
				qij = A[loop*n+i] ;
				ap[loop] += qij*p[i] ;
				ap[i] += qij*p[loop] ;
			}
			ap[loop] += p[loop]*A[loop*n+loop] ;
			pap += ap[loop]*p[loop] ;
		}
		if (pap < 0)
			printf("Warning : the matrix A is negative definite.\r\n") ;
		lambda = r1 / pap ;
		//update alpha and ri
		r0=r1 ;
		r1=0 ;
		nr=0 ;
		for (loop=0;loop<n;loop++)
		{
			x[loop] += lambda * p[loop] ;
			r[loop] -= lambda * ap[loop] ;
			r1 += r[loop]*r[loop] ;
			nr += fabs( r[loop] ) ;
		}
		nr = nr / (double)(n) ;
		index ++ ;
		//printf ("%u loop ... |r| = %f; obj = %f. \r\n", index, nr, OBJFUNC ) ;
	}

	free(r) ;
	free(ap) ;
	free(p) ;
	return 0 ;
}

