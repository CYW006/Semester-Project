#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "pref.h"

Pref_Settings * Create_Pref_Settings(Data_List * list) 
{
	Pref_Settings * settings = NULL ;
	char buf[2048] ;
	char * pstr ;
	int result ;
	FILE * fid ;
	unsigned int sz ;

	if (NULL == list)
		return NULL ;
	settings = (Pref_Settings *) malloc(sizeof(Pref_Settings)) ;
	if ( NULL != settings )
	{
		settings->kernel = DEF_KERNEL ;
		settings->pairs = list ;
		settings->cacheall = DEF_CACHEALL ;
		settings->ardon = DEF_ARDON ;
		//settings->kappa_a = DEF_KAPPA ;
#ifdef _PREF_EP
		settings->kappa_a = 1.0/(list->dimen) ;
#else
		settings->kappa_a = 1.0/sqrt(list->dimen) ;
#endif
		settings->kappa_o = DEF_KAPPA_O ;//* list->classes ;
		settings->kappa_m = DEF_KAPPA_M ;//+list->classes ;
		settings->kappa = NULL ;
		settings->noisevar = DEF_NOISEVAR ;
		settings->p = DEF_P ;
		settings->kfoldcv = DEF_KFOLDCV ;
		settings->invcov = NULL ;
		settings->alpha = NULL ;
		settings->regular = DEF_REGULAR ;
		settings->time = 0 ;
		// create testing file name 
		pstr = strstr(list->filename, "samples") ;
		if (NULL == pstr)
		{
			printf("The input file name should contain the string samples.\n") ;
			exit(1) ;
		}
		else
		{
			result = abs( list->filename - pstr ) ;
			strncpy (buf, list->filename, result ) ;
			buf[result] = '\0' ;
			strcat(buf, "train") ;
			strcat (buf, pstr+7) ;
			settings->trainfile = strdup(buf) ;
			
			result = abs( list->filename - pstr ) ;
			strncpy (buf, list->filename, result ) ;
			buf[result] = '\0' ;
			strcat(buf, "test") ;
			strcat (buf, pstr+7) ;
			settings->testfile = strdup(buf) ;
		}
		// ard
		if (1 == settings->ardon)
		{
			settings->kappa = (double*) malloc(list->dimen*sizeof(double)) ;
			if (NULL == settings->kappa)
			{
				printf("fail to malloc for ard.\n");
				exit(1);
			}
			strcpy(buf,list->filename) ;
			strcat( buf,".ard") ;
                        fid = fopen(buf,"r+t") ;
                        if (NULL != fid)
                        {
                                printf("Loading the initial ARD values in %s ...",buf) ;
                                sz = 0 ;
                                while (!feof(fid) && NULL!=fgets(buf,1024,fid) )
                                {
                                        if (strlen(buf)>1)
                                        {
                                                if (sz>=list->dimen)
                                                {
                                                        printf("Warning : ARD file is too long.\n") ;
                                                        sz = list->dimen-1 ;
                                                }
                                                settings->kappa[sz] = atof(buf) ;
                                                sz += 1 ;
                                        }
                                        else
                                                printf("Warning : blank line in ARD file.\n") ;
                                }
                                if (sz!=list->dimen)
                                {
                                        //default 0
                                	for (result=0;result<(int)list->dimen;result++)
                                        	settings->kappa[result] = 1.0/sqrt((double)list->dimen) ;
                                        printf(" RESET as default.\n") ;
                                }
                                else
                                        printf(" done.\n") ;
                                fclose(fid) ;
                        }
			else
			{
                                for (result=0;result<(int)list->dimen;result++)
                                        settings->kappa[result] = settings->kappa_a ;
			}	
			settings->number = settings->pairs->classes + settings->pairs->dimen ;
			printf("ARD kernel is used.\n") ;
		}
		else
			settings->number = 1 + settings->pairs->classes ; //(kappa_o) & kappa_a

		// intervals&thresholds
		settings->intervals = NULL ;
		settings->thresholds = NULL ;
		/*settings->intervals = (double*) calloc((list->classes-1),sizeof(double)) ;
		settings->thresholds = (double*) calloc((list->classes-1),sizeof(double)) ;
		if (NULL == settings->intervals || NULL == settings->thresholds)
		{
			printf("fail to malloc for intervals and thresholds.\n") ;
			exit(1) ;
		}
		settings->thresholds[0]=-settings->kappa_o/2.0+settings->kappa_o*(double)list->labelnum[0]/((double)list->count) ;
		for (result=1;result<(int)list->classes-1;result++)
			settings->intervals[result] = settings->kappa_o*(double)list->labelnum[result]/((double)list->count) ;
		for (result=1;result<(int)list->classes-1;result++)
			settings->thresholds[result] = settings->thresholds[result-1] + settings->intervals[result] ;
		*/
		//Alpha
		//settings->alpha = Create_Alphas (settings) ;
		//if (NULL == settings->alpha)
		//{
		//	printf("fail to create ALPHAs.\n") ;
		//	exit(1) ;
		//}
	}
	return settings ;
}

int Clear_Pref_Settings( Pref_Settings * settings) 
{
	if (NULL == settings)
		return 0 ;	
	if (NULL != settings->testfile)
		free(settings->testfile) ;
	if (NULL != settings->trainfile)
		free(settings->trainfile) ;
	if (NULL != settings->alpha)
		Clear_Alphas(settings->alpha, settings->pairs) ;
	if (NULL != settings->kappa)
		free(settings->kappa) ;
	if (NULL != settings->thresholds)
		free(settings->thresholds) ;
	if (NULL != settings->intervals )
		free(settings->intervals) ;
	if (NULL != settings->invcov)
		free(settings->invcov) ;
	Clear_Bfgs_FuncGrad(settings->bfgs) ;
	free(settings) ;
	return 0 ;
}

