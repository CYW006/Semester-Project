// main function 

// Chu Wei (C) Copyright 2004 at GATSBY

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "pref.h"
#define VERSION (1)

int main( int argc, char * argv[] )
{
	Data_List rawdatalist ;
	Pref_Settings * settings ;
	double linear = 0 ;
	double kappa_o = 0 ;
	double kappa = 0 ;
	double sigma = 0 ;
	double regular = -1 ;
	unsigned int sz = 0 ;
	unsigned int index = 0 ;
	double parameter = 0 ;
	char buf[2048]="" ;
	char filename[2048]="" ;

	printf("\n Gaussian Processes for Preference Learning - version 1.%d\n - Chu Wei Copyright(C) 2004 at Gatsby Unit\n\n", VERSION) ;

	if ( 1 == argc )
	{
		// display help 
		printf(" Usage : gpref [-K k] [-M m] [-O o][-R r] training_pairs samples_data\n") ;	
		printf(" - training_pairs: the data file containing training pairs\n") ;
		printf(" - samples_data: the data file containing sample features\n") ;
		printf(" - K k: set initial Kappa at k\n") ;
 		printf(" - S s: set initial Sigma^2 at s (default 1)\n") ;
		printf(" - O o: set initial kappa_O at o (default 1)\n") ;
		printf(" - G  : use Gaussian kernel (default Linear)\n\n") ;
		//printf(" - R r: set Regularization factor at r (default 1)\n\n") ;
		return 0 ;
	}
	else
	{
		sprintf(filename,"%s",argv[--argc]) ;
		if (argc>1)
			printf("Options:\n") ;
		do
		{
			strcpy(buf, argv[--argc]) ;
			sz = strlen(buf) ;
			//printf ("%s  %d\n", buf, sz) ;
			if ( '-' == buf[0] )
			{			
				for (index = 1 ; index < sz ; index++)
				{
					switch (buf[index])
					{
					case '-' :
						break ;
					case 'K' :						
						if (parameter > 0)
						{ 
							if (index + 1 == sz)
							{
								printf("  - initialize Kappa at the value %f.\n", parameter) ;
								kappa = parameter ;
								parameter = 0;
							}
						}
						break ;
					case 'O' :						
						if (parameter > 0)
						{ 
							if (index + 1 == sz)
							{
								printf("  - initialize kappa_O at the value %f.\n", parameter) ;
								kappa_o = parameter ;
								parameter = 0;
							}
						}
						break ;
					case 'S' :
						if (parameter > 0)
						{ 
							if (index + 1 == sz)
							{
								printf("  - initialize Sigma^2 at the value %f.\n", parameter) ;
								sigma = parameter ;
								parameter = 0;
							}
						}
						break ;
					case 'G' :
						printf("  - use GAUSSIAN kernel.\n") ;
						linear = 1 ;
						break ;
					case 'R' :						
						if (parameter >= 0)
						{ 
							if (index + 1 == sz)
							{
								printf("  - initialize Regularization factor at the value %f.\n", parameter) ;
								regular = parameter ;
								parameter = 0;
							}
						}
						break ;
					default :
						printf("  - %c is invalid.\n", buf[index]) ;
						break ;
					}
				}
			}
			else
				parameter = atof(buf) ;
		}
		while ( argc > 1 ) ;
		printf("\n") ;
	}

	Create_Data_List (&rawdatalist) ;
	if ( 1 == Pref_Loadfile (&rawdatalist, filename) )
	{
		printf("fail to load data from %s.\n\n",filename) ;
		exit(1) ;
	}
	settings = Create_Pref_Settings(&rawdatalist) ;
	if (NULL == settings)
	{
		printf("fail to generate ordinal settings.\n") ;
		exit(1) ;
	}
	else 
	{		
		if (kappa>0)
		{
				if (0==settings->ardon)		
					settings->kappa_a = kappa ;
				else
				{				
					for (index=0;index<settings->pairs->dimen;index++)
						settings->kappa[index] = kappa ;
				}
		}
		if (sigma>0)
			settings->noisevar = sigma ;
		if (kappa_o>0)
			settings->kappa_o = kappa_o ;
		if (regular>=0)
			settings->regular = regular ;
		if (linear>0.5)
			settings->kernel = GAUSSIAN ;
	}

	if ( 1 == Pref_Loadpair (&(rawdatalist.trainpair), settings->trainfile) )
	{
		printf("fail to load training pairs from %s.\n\n",settings->trainfile) ;
		exit(1) ;
	}

	settings->bfgs = Create_Bfgs_FuncGrad(settings->number) ;
	if (NULL == settings->bfgs)
	{
		printf("fail to create BFGSs.\n") ;
		exit(1) ;
	}
	Pref_Bfgs_Settings(settings,settings->bfgs) ;

	//Pref_MAP_Training (settings) ;

#ifdef _PREF_LP
    	Pref_LAPLACE_Training (settings) ;
#endif

	Pref_Prediction (settings) ;

	Dumping_Pref_Settings(settings) ;

	Clear_Pref_Settings(settings) ;
	Clear_Data_List (&rawdatalist) ;
	return 0 ;
}


