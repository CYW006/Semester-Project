/*******************************************************************************\

	smo_kernel.c in Sequential Minimal Optimization ver2.0
		
	calculates the Kernel.

	Chu Wei Copyright(C) National Univeristy of Singapore
	Create on Jan. 16 2000 at Control Lab of Mechanical Engineering 
	Update on Aug. 23 2001 

\*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/types.h> 
#include <sys/timeb.h>
#include "pref.h"

double Calc_Kernel( double * pi, double * pj, Pref_Settings * settings )
{
	long unsigned int dimen = 0 ;
	long unsigned int dimension = 0 ;
	double kernel = 0 ;

	if ( NULL == pi || NULL == pj || NULL == settings )
		return kernel ;
	
	if ( 1 == settings->ardon )
		if (NULL == settings->kappa)
			return kernel ;

	dimension = settings->pairs->dimen ;

	if ( POLYNOMIAL == settings->kernel )	
	{
		if ( 1 == settings->ardon )
		{
			for ( dimen = 0; dimen < dimension; dimen ++ )
				if ( pi[dimen]!=0 && pj[dimen]!=0 )
					kernel = kernel + settings->kappa[dimen] * pi[dimen] * pj[dimen] ;
		}
		else
		{
			for ( dimen = 0; dimen < dimension; dimen ++ )
				if ( pi[dimen]!=0 && pj[dimen]!=0 )
					kernel = kernel + settings->kappa_a * pi[dimen] * pj[dimen] ;	
		}
		if ((double) settings->p > 1.0)
			kernel = pow( (kernel), (double) settings->p ) ;
		else
			kernel = (kernel) ;
	}	
	else if ( LINEAR == settings->kernel )
	{
		if ( 1 == settings->ardon )
		{
			for ( dimen = 0; dimen < dimension; dimen ++ )
				if ( pi[dimen]!=0 && pj[dimen]!=0 )
					kernel = kernel + settings->kappa[dimen] * pi[dimen] * pj[dimen] ;
		}
		else
		{
			for ( dimen = 0; dimen < dimension; dimen ++ )
				if ( pi[dimen]!=0 && pj[dimen]!=0 )
					kernel = kernel + settings->kappa_a * pi[dimen] * pj[dimen] ;	
		}
		kernel = settings->kappa_o * kernel ; 
	}
	else if ( GAUSSIAN == settings->kernel )
	{
		if ( 1 == settings->ardon )
		{
			for ( dimen = 0; dimen < dimension; dimen ++ )
				//if ( pi[dimen]!=pj[dimen] )
					kernel = kernel + settings->kappa[dimen] * ( pi[dimen] - pj[dimen] ) * ( pi[dimen] - pj[dimen] ) ;
		}
		else
		{
			for ( dimen = 0; dimen < dimension; dimen ++ )
				//if ( pi[dimen]!=pj[dimen] )
					kernel = kernel + settings->kappa_a * ( pi[dimen] - pj[dimen] ) * ( pi[dimen] - pj[dimen] ) ;			
		}
		kernel = settings->kappa_o * exp ( -  kernel/2.0/settings->pairs->dimen ) ; 
	}
	else if ( USERDEFINED == settings->kernel )
	{		
		if ( 1 == settings->ardon )
		{
			for ( dimen = 0; dimen < dimension; dimen ++ )
			{
				if ( 0 == settings->pairs->featuretype[dimen] )
					kernel = kernel + settings->kappa[dimen] * ( pi[dimen] - pj[dimen] )*( pi[dimen] - pj[dimen] ) ;
				else if ( pi[dimen] != pj[dimen] )
					kernel = kernel + settings->kappa[dimen] * 1.0 ;
			}
		}
		else
		{
			for ( dimen = 0; dimen < dimension; dimen ++ )
			{
				if ( 0 == settings->pairs->featuretype[dimen] )
					kernel = kernel + settings->kappa_a * ( pi[dimen] - pj[dimen] )*( pi[dimen] - pj[dimen] ) ;
				else if ( pi[dimen] != pj[dimen] )
					kernel = kernel + settings->kappa_a * 1.0 ;
			}
		}
		kernel = settings->kappa_o * exp ( - kernel / 2.0 ) ;
	}

	return kernel ;	
}

double Calc_Covfun( double * pi, double * pj, Pref_Settings * settings )
{
	double kernel = 0 ;
	if ( NULL == pi || NULL == pj || NULL == settings )
		return kernel ;
	if ( 1 == settings->ardon )
	{
		if (NULL == settings->kappa)
		{
			printf("ARD parameters have not been initialized.\n") ;
			return kernel ;	
		}
	}
	// Call Calc_Kernel to initialize kernel
	kernel = Calc_Kernel( pi, pj, settings ) ;
	kernel = kernel + settings->kappa_m ; //plus bias term
	if (pi==pj)
		kernel += DEF_JITTER ;
	return kernel ;
}

double Calculate_Covfun( struct _Alphas * ai, struct _Alphas * aj, Pref_Settings * settings )
{
	double kernel = 0 ;
	int i, j ;

	if ( NULL == ai || NULL == aj || NULL == settings )
		return kernel ;

	if (settings->pairs->dimen<1)
	{
		printf("Warning : dimension is less than 1.\n") ;
		return kernel ;
	}

	if (1 == settings->cacheall)
	{
		// retrieve kernel values
		i = ai - settings->alpha ;
		j = aj - settings->alpha ;
		if (i >= j)
			return ai->kernels[j] ;
		else 
			return aj->kernels[i] ;
	}
	else
		return Calc_Covfun(ai->pair->point,aj->pair->point,settings) ;
}
// the end of smo_kernel.c 


