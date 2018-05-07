
// driver1.f -- translated by f2c (version of 23 April 1993  18:34:30).


#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include "lbfgsb.h"

#define BFGSPRINT ((long int)-1)
#define BFGSFACTR (1e6)
#define BFGSPGTOL (1e-5)
//#define BFGSNMAX ((long int)2700)
#define BFGSM ((long int)5)
#define BFGSMMAX (BFGSM)//17
#define	EVALLIMIT (200) 

/*                             DRIVER 1 */
/*     -------------------------------------------------------------- */
/*                SIMPLE DRIVER FOR L-BFGS-B (version 2.1) */
/*     -------------------------------------------------------------- */

/*        L-BFGS-B is a code for solving large nonlinear optimization */
/*             problems with simple bounds on the variables. */

/*        The code can also be used for unconstrained problems and is */
/*        as efficient for these problems as the earlier limited memory */
/*                          code L-BFGS. */

/*        This is the simplest driver in the package. It uses all the */
/*                    default settings of the code. */


/*     References: */

/*        [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited */
/*        memory algorithm for bound constrained optimization'', */
/*        SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208. */

/*        [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN */
/*        Subroutines for Large Scale Bound Constrained Optimization'' */
/*        Tech. Report, NAM-11, EECS Department, Northwestern University, */
/*        1994. */


/*          (Postscript files of these papers are available via anonymous */
/*           ftp to eecs.nwu.edu in the directory pub/lbfgs/lbfgs_bcm.) */

/*                              *  *  * */

/*        NEOS, November 1994. (Latest revision June 1996.) */
/*        Optimization Technology Center. */
/*        Argonne National Laboratory and Northwestern University. */
/*        Written by */
/*                           Ciyou Zhu */
/*        in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal. */

/*     NOTE: The user should adapt the subroutine 'timer' if 'etime' is */
/*           not available on the system.  An example for system */
/*           AIX Version 3.2 is available at the end of this driver. */

/*     ************** */
/* Subroutine */
extern int lbfgsb(integer *, integer *, doublereal *, 
	    doublereal *, doublereal *, integer *, doublereal *, doublereal *,
	     doublereal *, doublereal *, doublereal *, integer *, char *, 
	    integer *, char *, logical *, integer *, doublereal *, ftnlen, 
	    ftnlen);
	    
int lbfgsb_minfunc ( int (* Evaluate_FuncGrad)(void *, Bfgs_Settings *),
				  void * setting, Bfgs_Settings * bfgssetting ) 
{
    /* Local variables */
    char task[60];
    //doublereal f, *x, *g;
    integer m, n;
    doublereal *l=NULL, *u=NULL, factr;
    char csave[60];
    doublereal dsave[29];
    integer isave[44];
    logical lsave[4];
    doublereal pgtol, *wa;
    integer iprint, *nbd, *iwa;
/*
	// bin file
	struct stat s ;
	int ms_fd ;
	FILE * fid ;
	long unsigned int i, j ;
	double t=0 ;
*/

	if ( NULL == bfgssetting )
		return 1 ;

/*     This simple driver demonstrates how to call the L-BFGS-B code to */
/*       solve a sample problem (the extended Rosenbrock function */
/*       subject to bounds on the variables). The dimension n of this */
/*       problem is variable. */
/*        nmax is the dimension of the largest problem to be solved. */
/*        mmax is the maximum number of limited memory corrections. */
/*     Declare the variables needed by the code. */
/*       A description of all these variables is given at the end of */
/*       the driver. */
/*     Declare a few additional variables for this sample problem. */
/*     We wish to have output at every iteration. */

	// copy bfgs settings
    iprint = BFGSPRINT ; // no print
/*     We specify the tolerances in the stopping criteria. */
    factr = BFGSFACTR ; //1e7;
    pgtol = BFGSPGTOL ; //1e-5;
/*     We specify the dimension n of the sample problem and the number */
/*        m of limited memory corrections stored.  (n and m should not */
/*        exceed the limits nmax and mmax respectively.) */
    m = BFGSM ;
    n = bfgssetting->number ;
/*     We now provide nbd which defines the bounds on the variables: */
/*                    l   specifies the lower bounds, */
/*                    u   specifies the upper bounds. */
/*     First set bounds on the odd-numbered variables. */

    //g=(doublereal*)malloc(n*sizeof(double)) ;
    //x=(doublereal*)malloc(n*sizeof(double)) ;
    //l=(doublereal*)malloc(n*sizeof(double)) ;
    //u=(doublereal*)malloc(n*sizeof(double)) ;
    nbd=(integer*)calloc(n,sizeof(integer)) ;
    iwa=(integer*)malloc(3*n*sizeof(integer)) ;
    wa=(double*)malloc((4*n+2*n*BFGSMMAX+12*BFGSMMAX+12*BFGSMMAX*BFGSMMAX)*sizeof(double)) ;
/*	
	fid = fopen("wa.bin","w+b") ;
	for (i=0;i<12*BFGSMMAX+12*BFGSMMAX*BFGSMMAX;i++)
		fwrite( &t, sizeof(double), 1, fid ) ;
	for (i=0;i<n;i++)
		for (j=0;j<4+2*BFGSMMAX;j++)
			fwrite( &t, sizeof(double), 1, fid ) ;
	fclose(fid) ;
	// mount wa
	if ((ms_fd = open("wa.bin", O_RDWR)) == -1)
	{
		printf("cannot open %s\n", "wa.bin"); 
		exit(1);
	}
	fstat(ms_fd, &s);
	printf("file size %ld\n", s.st_size) ;
	assert(s.st_size==(4*n+2*n*BFGSMMAX+12*BFGSMMAX+12*BFGSMMAX*BFGSMMAX)*sizeof(double)) ;
	wa = (double*) mmap(0, s.st_size, PROT_WRITE, MAP_SHARED, ms_fd, 0) ;
	if (MAP_FAILED==wa)
	{
		printf("fail to map wa.bin.\n") ;
		abort() ;
	}
	// wa mounted
*/
	if (m>BFGSMMAX)
		abort() ;

/*  We now define the starting point. */

    printf("bfgs with %ld variables\n",n) ;

/*  We start the iteration by initializing task. */

    s_copy(task, "START", 60L, 5L);
/*        ------- the beginning of the loop ---------- */
L111:
/*     This is the call to the L-BFGS-B code. */
    lbfgsb(&n, &m, bfgssetting->x, l, u, nbd, &bfgssetting->fx, bfgssetting->gx, &factr, &pgtol, wa, iwa, task, &
	    iprint, csave, lsave, isave, dsave, 60L, 60L);
    if (s_cmp(task, "FG", 2L, 2L) == 0) {
/*        the minimization routine has returned to request the */
/*        function f and gradient g values at the current x. */
/*        Compute function value f for the sample problem. */
/* Computing 2nd power */

	// calculate fx and gx at the POINT x 
	if ( 1 == (*Evaluate_FuncGrad)(setting, bfgssetting) ) 
	{
		printf( "\r\nFATAL ERROR : Evaluate_FuncGrad failed in BFGS.\r\n" ) ;
		abort() ;
	}
	
/*          go back to the minimization routine. */
	goto L111;
    }

    	if (s_cmp(task, "NEW_X", 5L, 5L) == 0) 
	{
		bfgssetting->iternum = isave[29] ;
		printf("%2d: obj f = %.2f	|proj g| = %.2f  	with %d evaluations\n",
			(int)isave[29], bfgssetting->fx, dsave[12], (int)isave[33] ) ;
		if (isave[33] >= EVALLIMIT && EVALLIMIT>0) 
		{
			s_copy(task, "STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT", 60L, 52L);
			printf ("Stop: the number of evaluations exceeds limit.\n") ;
		}
		if (dsave[12] <= ( fabs(bfgssetting->fx) + 1.) * 1e-5) 
		{
			s_copy(task, "STOP: THE PROJECTED GRADIENT IS SUFFICIENTLY SMALL", 60L, 50L);
			printf("Stop: the projected gradient is sufficiently small.\n") ;
		}		
		/*if ( fabs(dsave[1]-f) < fabs(dsave[1])*1e-6 && isave[29] > 2 ) 
		{
			s_copy(task, "STOP: THE DESCENT ON FUNCTION IS SUFFICIENTLY SMALL", 60L, 50L);
			printf("Stop: the descent on function is sufficiently small.\n") ;
		}*/
		goto L111;
    	}
/*     the minimization routine has returned with a new iterate, */
/*     and we have opted to continue the iteration. */
/*     ---------- the end of the loop ------------- */
/*     If task is neither FG nor NEW_X we terminate execution. */	

	bfgssetting->iternum = isave[29] ;
	//free(g);
	//free(x);
	//free(l);
	//free(u);
	free(nbd);
	free(iwa);
/*
	// unmount
	munmap(wa, s.st_size) ;
	close(ms_fd) ;
*/
	free(wa);
	return 0 ;
}
#ifdef __cplusplus
	}
#endif
