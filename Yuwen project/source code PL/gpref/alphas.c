// main function 

// softmax for multi-classification

// Chu Wei (C) Copyright 2004

#include <stdio.h>
#include <stdlib.h>
#include "pref.h"

Alphas * Create_Alphas (Pref_Settings * settings)
{
	Alphas * alpha = NULL ;
	Data_Node * node ;
	Pref_Node * pair ;
	int i = 0, u, v ;
	int serial = 0 ;
	int j ;
	if (NULL == settings->pairs)
		return alpha ;

	alpha = (Alphas *) malloc(settings->pairs->count*sizeof(Alphas)) ;
	if (NULL == alpha)
	{
		printf("fail to malloc alpha.\n") ;
		return alpha ;
	}

	// initialize
	node = settings->pairs->front ;
	while (NULL != node)
	{
		(alpha+i)->pair = node ;
		(alpha+i)->p_cache = NULL ;
		(alpha+i)->pair->serial = 0 ;
		(alpha+i)->pair->fold = 1 ;
		(alpha+i)->kernels = (double *)malloc((i+1)*sizeof(double)) ;		
		(alpha+i)->postcov = (double *)malloc((i+1)*sizeof(double)) ;
		if (NULL == (alpha+i)->kernels || NULL == (alpha+i)->postcov)
		{
			printf("fail to malloc alpha->kernels.\n") ;
			exit(1) ;
		}
		for (j=0;j<=i;j++)
		{
			(alpha+i)->kernels[j] = Calc_Covfun((alpha+i)->pair->point,(alpha+j)->pair->point,settings) ; 
			(alpha+i)->postcov[j] = (alpha+i)->kernels[j] ;
		}
		(alpha+i)->alpha = 0 ;
		(alpha+i)->beta = 0 ;
		(alpha+i)->nu = 0 ;
		(alpha+i)->loomean = 0 ;
		(alpha+i)->loovar = 0 ;
		(alpha+i)->z1 = 0 ;
		(alpha+i)->z2 = 0 ;	
		(alpha+i)->hnew = 0 ; // new posterior mean
		(alpha+i)->mnew = 0 ; // new individual mean
		(alpha+i)->pnew = 0 ; // new individual variance
		(alpha+i)->snew = 0 ; // new individual amplitude
		
		// clear node
		node->epamp = 1 ;
		node->epinvvar = 0 ;
		node->epmean = 0 ;
		node->postmean = 0 ;
		node->weight = 0 ;
		i+=1 ;
		node = node -> next ;
	}
		
	// mark the training points
	pair = settings->pairs->trainpair.front ;
	settings->pairs->train = 0 ;
	while (NULL != pair)
	{
		u = pair->u ;
		v = pair->v ;
		if ((alpha+(u-1))->pair->fold>0)
		{
			(alpha+(u-1))->pair->fold = -1 ;
			settings->pairs->train += 1 ;
		}
		if ((alpha+(v-1))->pair->fold>0)
		{
			(alpha+(v-1))->pair->fold = -1 ;
			settings->pairs->train += 1 ;
		}
		pair = pair->next ;
	}
	//	
	node = settings->pairs->front ;
	i=0 ;
	while (NULL != node)
	{
		if ((alpha+i)->pair->fold<0)
		{
			serial += 1 ;
			(alpha+i)->pair->serial = serial ;
		}
		else
			(alpha+i)->pair->serial = 0 ;
		node = node->next ;
		i+=1 ;
	}
	return alpha ;
}

int Clear_Alphas (Alphas * alpha, Data_List * list)
{
	Data_Node * node ;	
	int i = 0 ;

	if (NULL == list || NULL == alpha)
		return 1 ;
	node = list->front ;
	while (NULL != node)
	{
		if (NULL != (alpha+i)->p_cache)
			free((alpha+i)->p_cache) ;
		if (NULL != (alpha+i)->kernels)
			free((alpha+i)->kernels) ;
		if (NULL != (alpha+i)->postcov)
			free((alpha+i)->postcov) ;
		i+=1 ;
		node = node -> next ;
	}
	free(alpha) ;
	alpha = NULL ;
	return 0 ;
}
