/*******************************************************************************************\

	bfgs_settings.c in Sequential Minimal Optimization ver2.0 
	
	implements the functions of creating and clearing bfgs_Setting structure.

	Chu Wei Copyright(C) National Univeristy of Singapore
	Create on Jan. 16 2000 at Control Lab of Mechanical Engineering 
	Update on Aug. 23 2001 

\*******************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <sys/types.h> 
#include <sys/timeb.h>
#include "pref.h"


/*******************************************************************************************\

	BOOL Clear_bfgs_FuncGrad ( bfgs_FuncGrad * funcgrad )  
	
	purpose: to clear a bfgs_FuncGrad structure. 
	input:  the pointer to the bfgs_FuncGrad structure. 
	output: TRUE or FALSE

\*******************************************************************************************/

void Clear_Bfgs_FuncGrad ( bfgs_FuncGrad * funcgrad )
{
	if ( NULL != funcgrad )
	{
		if (NULL != funcgrad->gx)
			free (funcgrad->gx) ;
		if (NULL != funcgrad->x)
			free (funcgrad->x) ;
		free (funcgrad) ;
		funcgrad = NULL ;
	}
}

/*******************************************************************************************\

	bfgs_FuncGrad * Create_bfgs_FuncGrad ( unsigned int number )  
	
	purpose: to create and initialize a structure of bfgs_FuncGrad. 
	input:  the number of all adjustable parameters. 
	output: the pointer of bfgs_FuncGrad

\*******************************************************************************************/

bfgs_FuncGrad * Create_Bfgs_FuncGrad ( unsigned int number )
{	
	bfgs_FuncGrad * funcgrad = NULL ;
	unsigned int i = 0 ;

	if (number <= 0 )
		return NULL ;

	if ( NULL == ( funcgrad = (bfgs_FuncGrad *) malloc(sizeof(bfgs_FuncGrad)) ) )
		return NULL ;

	funcgrad->x = NULL ;
	funcgrad->gx = NULL ;
	funcgrad->var_number = number ;
	funcgrad->norm2gx = 0 ;
	funcgrad->maskednorm = 0 ;
	funcgrad->fx = 0 ;
	funcgrad->iternum = 0 ;

	if ( NULL == ( funcgrad->x = (double *) malloc(number*sizeof(double)) ) )
	{
		Clear_Bfgs_FuncGrad ( funcgrad ) ;
		return NULL ;
	}
	if ( NULL == ( funcgrad->gx = (double *) malloc(number*sizeof(double)) ) )
	{
		Clear_Bfgs_FuncGrad ( funcgrad ) ;
		return NULL ;
	}
	for ( i=0;i<number;i++)
	{
		funcgrad->x[i] = 0 ;
		funcgrad->gx[i] = 0 ;
	}	
	return funcgrad ;
}


/*******************************************************************************************\

	BOOL bfgs_smo_Settings ( smo_Settings * setting, bfgs_FuncGrad * funcgrad )  
	
	purpose: change smo parameters by copying funcgrad->X into the smo_Settings structure 
			and be called by bfgs_training only to change parameter settings.
	input:  the pointer to structure of smo_Settings and bfgs_FuncGrad 
	output: TRUE or FALSE

\*******************************************************************************************/

int Bfgs_Pref_Settings ( Pref_Settings * settings, bfgs_FuncGrad * funcgrad ) 
{
	unsigned int i = 0 ;

	if ( NULL == settings || NULL == funcgrad )
	{
		printf("\r\nFATAL ERROR : input pointer is NULL in bfgs_smo_Settings.\r\n") ;
		return 1 ;
	}
	if (PREFERENCE == settings->pairs->datatype)
	{
		/*if (funcgrad->x[0]>1000)
			funcgrad->x[0]=1000 ;
		if (funcgrad->x[0]<0.001)
			funcgrad->x[0]=0.001 ;*/

		settings->noisevar = 1.0/funcgrad->x[0] ;
		//settings->kappa_o = exp( funcgrad->x[funcgrad->var_number-1] ) ;
		/*settings->thresholds[0] = ( funcgrad->x[1] ) ;		
		for (i=1;i<settings->pairs->classes-1;i++) // intervals
		{
			settings->intervals[i] = exp( funcgrad->x[1+i] ) ;
			if (settings->intervals[i]>6.0*sqrt(settings->noisevar))
			{
				settings->intervals[i] = 6.0*sqrt(settings->noisevar) ;
				funcgrad->x[1+i] = log(settings->intervals[i]) ;				
			}
			settings->thresholds[i] = settings->thresholds[i-1] + settings->intervals[i] ;
		}*/
		if ( 1 == settings->ardon )
		{
			for ( i=0; i<settings->pairs->dimen; i++)
			{
				settings->kappa[i] = exp( funcgrad->x[settings->pairs->classes+i] ) ;
				if (settings->kappa[i]>50)
                                {
                                        settings->kappa[i]=50 ;
                                        funcgrad->x[settings->pairs->classes+i]=log(settings->kappa[i]) ;
                                }
                                else if (settings->kappa[i]<10e-10)
                                {
                                        settings->kappa[i]=10e-10 ;
                                        funcgrad->x[settings->pairs->classes+i]=log(settings->kappa[i]) ;
                                }
			}
		}
		else
			settings->kappa_a = exp( funcgrad->x[settings->pairs->classes] ) ;
	}
	else
	{
		printf("\r\nFATAL ERROR : wrong DATATYPE in bfgs_smo_Settings.\r\n") ;
		return 1 ;
	}
	return 0 ;
}

int Pref_Bfgs_Settings ( Pref_Settings * settings, bfgs_FuncGrad * funcgrad ) 
{
	unsigned int i = 0 ;

	if ( NULL == settings || NULL == funcgrad )
	{
		printf("\r\nFATAL ERROR : input pointer is NULL in bfgs_smo_Settings.\r\n") ;
		return 1 ;
	}
	if (PREFERENCE == settings->pairs->datatype)
	{
		funcgrad->x[0] = 1.0/(settings->noisevar) ;
		//funcgrad->x[funcgrad->var_number-1] = log(settings->kappa_o) ;
		/*funcgrad->x[1] = settings->thresholds[0] ;		
		for (i=1;i<settings->pairs->classes-1;i++) // intervals
			funcgrad->x[1+i] = log(settings->intervals[i]) ;*/
		if ( 1 == settings->ardon )
		{
			for ( i=0; i<settings->pairs->dimen; i++ )
				funcgrad->x[settings->pairs->classes+i] = log(settings->kappa[i]) ;
		}
		else
			funcgrad->x[settings->pairs->classes] = log(settings->kappa_a) ;
	}
	else
	{
		printf("\r\nFATAL ERROR : wrong DATATYPE in bfgs_smo_Settings.\r\n") ;
		return 1 ;
	}
	return 0 ;
}

/*******************************************************************************************\

	BOOL Duplicate_bfgs_FuncGrad( bfgs_FuncGrad * destination, bfgs_FuncGrad * source )
	
	purpose: copy all the elements of bfgs_FuncGrad from source to destination. 				
	input:  the pointer of bfgs_FuncGrad, destination and source. 
	output: TRUE or FALSE

\*******************************************************************************************/

int Duplicate_Bfgs_FuncGrad( bfgs_FuncGrad * destination, bfgs_FuncGrad * source )
{
	unsigned int i = 0 ;

	if ( NULL == destination || NULL == source )
	{
		printf("\r\nFATAL ERROR : in Duplicate_bmr_FuncGrad.\r\n") ;
		return 1 ;
	}
	if ( destination->var_number != source->var_number )
	{
		printf("\r\nFATAL ERROR : in Duplicate_bmr_FuncGrad.\r\n") ;
		return 1 ;
	}

	destination->var_number = source->var_number ;
	destination->fx = source->fx ;
	destination->norm2gx = source->norm2gx ;
	destination->maskednorm = source->maskednorm ;
	for ( i=0; i<source->var_number; i++ )
	{
		destination->gx[i] = source->gx[i] ;
		destination->x[i] = source->x[i] ;
	}
	return 0 ;
}

// end of bfgs_settings.c

