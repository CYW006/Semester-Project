#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include "pref.h"
#include "neal_matrix.h"

#define UPPERFUNC (10.0)

int Compute_MAP_Weights (Pref_Settings * settings) 
{
	double sigma, z, n, phi ;
	unsigned int i,u,v ;
	Pref_Node * node ;
#ifdef _PREF_DEBUG
	unsigned int j ;
	double * f ;
	double * w ;
	double * t ;
	double error = 0 ;
#endif

	assert(NULL!=settings) ;
	assert(settings->noisevar>0) ;

	sigma = sqrt(2.0*settings->noisevar) ; // sigma^\star

	i=0 ;
	node = settings->pairs->trainpair.front ;
	while (NULL != node)
	{
		// who is old function
		// update alpha_old
		u = (settings->alpha+(node->u-1))->pair->serial ;
		v = (settings->alpha+(node->v-1))->pair->serial ;
		assert(u<=node->u&&u<=settings->pairs->train) ;
		assert(v<=node->v&&v<=settings->pairs->train) ;
		z = (settings->alpha+(node->u-1))->pair->postmean 
			- (settings->alpha+(node->v-1))->pair->postmean ; // f(u_k)-f(v_k)
		z = z / sigma ;
		n = normpdf(z) ;
		phi = 0.5 + normcdf(0,z) ;
		(settings->alpha+(node->u-1))->pair->weight += n/phi/sigma ;
		(settings->alpha+(node->v-1))->pair->weight -= n/phi/sigma ;		
		node = node -> next ;
		i+=1 ;
	}
	assert(i==settings->pairs->trainpair.count) ;
#ifdef _PREF_DEBUG
	for (i=0;i<settings->pairs->count;i++)
		printf("w(%d)=%.3f\n",i,(settings->alpha+i)->pair->weight) ;
	f = (double *)malloc(settings->pairs->train*sizeof(double)) ;
	w = (double *)malloc(settings->pairs->train*sizeof(double)) ;
	t = (double *)malloc(settings->pairs->train*sizeof(double)) ;
	if (NULL != settings->invcov) 
		free (settings->invcov) ;
	settings->invcov = (double *)malloc(settings->pairs->train*settings->pairs->train*sizeof(double)) ;
	u=0 ;
	for (i=0;i<settings->pairs->count;i++)
	{
		if ((settings->alpha+i)->pair->fold < 0)
		{
			assert((settings->alpha+i)->pair->serial>0) ;
			v=0 ;
			for (j=0;j<i;j++)
			{
				if ((settings->alpha+j)->pair->fold < 0)
				{
					settings->invcov[u*settings->pairs->train+v] = 
						Calculate_Covfun(settings->alpha+i,settings->alpha+j,settings) ;
					settings->invcov[v*settings->pairs->train+u] = 
						settings->invcov[u*settings->pairs->train+v] ;
					v+=1 ;
				}
			}
			f[u] = (settings->alpha+i)->pair->postmean ; // current f
			w[u] = (settings->alpha+i)->pair->weight ;
			settings->invcov[u*settings->pairs->train+u] = Calculate_Covfun(settings->alpha+i,settings->alpha+i,settings) ;
			u+=1 ;
		}
	}
	assert(u==settings->pairs->train) ;

	for (i=0;i<settings->pairs->train;i++)
		printf("w(%d)=%.3f\n",i,w[i]) ;


	for (i=0;i<settings->pairs->train;i++)
	{
		t[i] = 0 ;
		for (j=0;j<i;j++)
		{
			t[i] += settings->invcov[i*settings->pairs->train+j]*w[j] ;			
			t[j] += settings->invcov[j*settings->pairs->train+i]*w[i] ;	
		}
		t[i] += settings->invcov[i*settings->pairs->train+i]*w[i] ;	
	}
	for (i=0;i<settings->pairs->train;i++)
	{
		error += fabs(f[i]-t[i]) ;
	}
	printf("distance : %f.\n", error) ;
	free(t) ;
	free(w) ;
	free(f) ;
#endif
	return 0 ;
}

double Pref_MAP_Update(Pref_Settings * settings)
{
	double error = 0 ;
	//unsigned int cnt = 0 ;
	double sigma ;
	double * f ;
	double * a ;
	double * lambda ;
	unsigned int i, j, u=0, v=0 ;
	double z1, phi1, n1 ;
	//double func = 0 ;
	double step = 1.0 ;
	Pref_Node * node ;
	double * t1 ;
	double * Q ;
	double * df ;
	double err ;

	if (NULL == settings)
	{
		printf("Input pointer is NULL.\n") ;
		return  0 ;
	}
	if (settings->noisevar<=0)
		return 0 ;

	sigma = sqrt(2.0*settings->noisevar) ; // sigma^\star

	Q = (double *)malloc(settings->pairs->train*settings->pairs->train*sizeof(double)) ; // Sigma	
	lambda = (double *)calloc(settings->pairs->train*settings->pairs->train,sizeof(double)) ; // Sigma	
	f = (double *)malloc(settings->pairs->train*sizeof(double)) ;
	t1 = (double *)malloc(settings->pairs->train*sizeof(double)) ;
	df = (double *)malloc(settings->pairs->train*sizeof(double)) ;
	a = (double *)calloc(settings->pairs->train,sizeof(double)) ;

	for (i=0;i<settings->pairs->count;i++)
	{
		(settings->alpha+i)->pair->weight = 0 ; // first order
		if ((settings->alpha+i)->pair->fold < 0)
		{
			assert((settings->alpha+i)->pair->serial>0) ;			
			f[u] = (settings->alpha+i)->pair->postmean ; // current f
			u+=1 ;
		}
	}
	assert(u==settings->pairs->train) ;

	
	/*for (i=0;i<settings->pairs->train;i++)
	{
		for (j=0;j<settings->pairs->train;j++)
			printf("%.2f ", settings->invcov[i*settings->pairs->train+j]) ;
			printf("\n") ;
	}*/

	// prepare w_i
	i=0 ;
	node = settings->pairs->trainpair.front ;
	while (NULL != node)
	{
		// who is old function
		// update alpha_old
		u = (settings->alpha+(node->u-1))->pair->serial ;
		v = (settings->alpha+(node->v-1))->pair->serial ;
		assert(u<=node->u&&u<=settings->pairs->train) ;
		assert(v<=node->v&&v<=settings->pairs->train) ;
		z1 = f[u-1] - f[v-1] ; // f(u_k)-f(v_k)
		z1 = z1 / sigma ;
		n1 = normpdf(z1) ;
		phi1 = 0.5 + normcdf(0,z1) ;
		//(settings->alpha+(u-1))->pair->weight += n1/phi1/sigma ;		
		//(settings->alpha+(v-1))->pair->weight += n1/phi1/sigma ;
		a[u-1]-=n1/phi1/sigma ;		
		a[v-1]+=n1/phi1/sigma ;
		lambda[(u-1)*settings->pairs->train+(u-1)] += 
			( (n1*n1)/(phi1*phi1)+z1*n1/phi1)/sigma/sigma ;
		lambda[(v-1)*settings->pairs->train+(v-1)] += 
			( (n1*n1)/(phi1*phi1)+z1*n1/phi1)/sigma/sigma ;
		lambda[(u-1)*settings->pairs->train+(v-1)] -= 
			( (n1*n1)/(phi1*phi1)+z1*n1/phi1)/sigma/sigma ;
		lambda[(v-1)*settings->pairs->train+(u-1)] -= 
			( (n1*n1)/(phi1*phi1)+z1*n1/phi1)/sigma/sigma ;

		
		/*(settings->alpha+i)->alpha = (settings->alpha+i)->pair->epamp ;
		(settings->alpha+i)->pair->epmean = (settings->alpha+i)->pair->postmean ;
	
		n1 = 0 ;
		n2 = 0 ;
		z1 = 0 ;
		z2 = 0 ;
		phi1 = 1 ;
		phi2 = 0 ;

		if (1 == (settings->alpha+i)->pair->target)
		{
			z1 = (settings->thresholds[(settings->alpha+i)->pair->target-1]-(settings->alpha+i)->pair->epmean)/sigma ;
			n1 = normpdf(z1) ;
			phi1 = 0.5 + normcdf(0,z1) ;
			dphi = phi1-phi2 ;
		}
		else if (settings->pairs->classes == (settings->alpha+i)->pair->target)
		{
			z2 = (settings->thresholds[(settings->alpha+i)->pair->target-2]-(settings->alpha+i)->pair->epmean)/sigma ;
			n2 = normpdf(z2) ;
			phi2 = 0.5 + normcdf(0,z2) ;
			dphi = phi1-phi2 ;
		}
		else
		{
			z1 = (settings->thresholds[(settings->alpha+i)->pair->target-1]-(settings->alpha+i)->pair->epmean)/sigma ;
			z2 = (settings->thresholds[(settings->alpha+i)->pair->target-2]-(settings->alpha+i)->pair->epmean)/sigma ;			
			n1 = normpdf(z1) ;
			n2 = normpdf(z2) ;
			dphi = normcdf(z2,z1);
		}

		if (dphi<EPS)
		{
			if (n1-n2<EPS)
			{
				func -= log(EPS) ;
				step = 0.01 ;		
			}
			else
			{
				func -= log(EPS) ;
				step = 0.1 ;
			}

			if (1 == (settings->alpha+i)->pair->target)
			{
				(settings->alpha+i)->pair->weight = -z1/sigma ;
				(settings->alpha+i)->pair->epinvvar = 1/sigma/sigma ;
			}
			else if (settings->pairs->classes == (settings->alpha+i)->pair->target)
			{
				(settings->alpha+i)->pair->weight = z2/sigma ;
				(settings->alpha+i)->pair->epinvvar = 1/sigma/sigma ;
			}
			else
			{
				if ((n1-n2)>=0)
				{
					(settings->alpha+i)->pair->weight = -(z1*exp(-0.5*z1*z1+0.5*z2*z2)-z2)/(exp(-0.5*z1*z1+0.5*z2*z2)-1.0)/sigma ;
					(settings->alpha+i)->pair->epinvvar = 1/sigma/sigma ;
					//(settings->alpha+i)->pair->epinvvar = 1/sigma/sigma 
					//	+ (settings->alpha+i)->pair->weight*(settings->alpha+i)->pair->weight
					//	- (z1*z1*exp(-0.5*z1*z1+0.5*z2*z2)-z2*z2)/(exp(-0.5*z1*z1+0.5*z2*z2)-1.0)/sigma/sigma ;
				}
				else
				{
					//printf("n1 < n2 ;\n") ;
					(settings->alpha+i)->pair->weight = 0 ;
					(settings->alpha+i)->pair->epmean = 0 ;
					(settings->alpha+i)->pair->postmean = 0 ;
					(settings->alpha+i)->pair->epinvvar = 1/sigma/sigma ;
				}
			}
#ifdef _ORDINAL_DEBUG
			//printf("%u approximation, w=%f, p=%f\n",i,(settings->alpha+i)->pair->weight,(settings->alpha+i)->pair->epinvvar) ;
#endif
		}
		else
		{
			(settings->alpha+i)->pair->weight = (n1-n2)/(dphi)/sigma ;
			(settings->alpha+i)->pair->epinvvar = (settings->alpha+i)->pair->weight*(settings->alpha+i)->pair->weight 
				+ (z1*n1-z2*n2)/(dphi)/sigma/sigma ;
			// update func 			
			func -= log(dphi) ;
		}

		(settings->alpha+i)->pair->weight = -(settings->alpha+i)->pair->weight ;

		if ((settings->alpha+i)->pair->epinvvar>1.0/sigma/sigma)
		{
			//printf("error : %u, %f\n", i, (settings->alpha+i)->pair->epinvvar) ;
			(settings->alpha+i)->pair->epinvvar = 1.0 - EPS*EPS ;
		}
		
		if ((settings->alpha+i)->pair->epinvvar<0)
		{
			//printf("error : %u, %f\n", i, (settings->alpha+i)->pair->epinvvar) ;
			(settings->alpha+i)->pair->epinvvar = EPS*EPS ;
		}*/
		node = node -> next ;
		i+=1 ;
	}
	assert(i==settings->pairs->trainpair.count) ;

	for (i=0;i<settings->pairs->train;i++)
	{
		t1[i] = 0 ;
		for (j=0;j<i;j++)
		{
			t1[i] += settings->invcov[i*settings->pairs->train+j]*f[j] ;			
			t1[j] += settings->invcov[j*settings->pairs->train+i]*f[i] ;	
		}
		t1[i] += settings->invcov[i*settings->pairs->train+i]*f[i] ;	
	}
	for (i=0;i<settings->pairs->train;i++)
	{
		a[i] += t1[i] ;
		//printf("delta a(%d)=%f\n",i,a[i]) ;
	}

	// Sigma*Lambda
	for (i=0;i<settings->pairs->train;i++)
	{
		for (j=0;j<i;j++)
		{
			Q[i*settings->pairs->train+j] = settings->invcov[i*settings->pairs->train+j] + lambda[i*settings->pairs->train+j] ; 
			Q[j*settings->pairs->train+i] = Q[i*settings->pairs->train+j] ; 
		}
		Q[i*settings->pairs->train+i] = settings->invcov[i*settings->pairs->train+i] + lambda[i*settings->pairs->train+i] ; 
	}

	conjugate_gradient( Q, a, df, settings->pairs->train ) ;

/*
	if (0 == cholesky(settings->invcov,settings->pairs->train,&temp))
		printf("Fatal Error: Sigma+1/Pi is not positive definite.\n") ;
	if (0 == inverse_from_cholesky(settings->invcov, t1, t2, settings->pairs->train) )		
		printf("Fatal Error: Sigma+1/Pi is not positive definite.\n") ;

	for (i=0;i<settings->pairs->train;i++)
	{
		t1[i] = 0 ;
		for (j=0;j<i;j++)
		{
			t1[i] += settings->invcov[i*settings->pairs->train+j]*a[j] ;			
			t1[j] += settings->invcov[j*settings->pairs->train+i]*a[i] ;	
		}
		t1[i] += settings->invcov[i*settings->pairs->train+i]*a[i] ;	
	}
*/
	err = 0 ;
	for (i=0;i<settings->pairs->train;i++)
		err += df[i] * df[i] ;

	if (err/(double)settings->pairs->train>1.0) 
		step = (double)settings->pairs->train / err ;

	j=0 ;
	for (i=0;i<settings->pairs->count;i++)
	{
		if ((settings->alpha+i)->pair->fold < 0)
		{
		
		(settings->alpha+i)->pair->postmean -= step*df[j] ; 
		//printf("delta f(%d)=%f\n",i, t1[j]) ;	
		/*if (fabs((settings->alpha+j)->pair->postmean)>UPPERFUNC)
		{
//#ifdef _ORDINAL_DEBUG
			printf("%u - new %f --- old %f ---- w %f \n",i, (settings->alpha+j)->pair->postmean,(settings->alpha+j)->pair->epmean,(settings->alpha+j)->pair->weight) ;
//#endif
			//step = (UPPERFUNC-fabs((settings->alpha+i)->pair->epmean))/fabs(t1[i]) ;
			(settings->alpha+j)->pair->postmean = UPPERFUNC ;			
		}*/
		if (error<fabs(df[j]))		
			error = fabs(df[j]) ;
		j+=1 ;
		}
	}
#ifdef _PREF_DEBUG
	
	u=0 ;
	for (i=0;i<settings->pairs->count;i++)
	{
		(settings->alpha+i)->pair->weight = 0 ; // first order
		if ((settings->alpha+i)->pair->fold < 0)
		{
			assert((settings->alpha+i)->pair->serial>0) ;		
			f[u] = (settings->alpha+i)->pair->postmean ; // current f
			u+=1 ;
		}
	}
	assert(u==settings->pairs->train) ;
	i=0 ;
	node = settings->pairs->trainpair.front ;
	while (NULL != node)
	{
		// who is old function
		// update alpha_old
		u = (settings->alpha+(node->u-1))->pair->serial ;
		v = (settings->alpha+(node->v-1))->pair->serial ;
		assert(u<=node->u&&u<=settings->pairs->train) ;
		assert(v<=node->v&&v<=settings->pairs->train) ;
		z1 = f[u-1] - f[v-1] ; // f(u_k)-f(v_k)
		z1 = z1 / sigma ;
		n1 = normpdf(z1) ;
		phi1 = 0.5 + normcdf(0,z1) ;
		func -= log(phi1) ;
		node = node -> next ;
	}
	for (i=0;i<settings->pairs->train;i++)
	{
		for (j=0;j<i;j++)
		{
			func += settings->invcov[i*settings->pairs->train+j] * f[i] * f[j] ;
		}
		func += 0.5 * settings->invcov[i*settings->pairs->train+i] * f[i] * f[i] ;
	}
	printf("FUNCTIONAL : %f \n", func) ;
#endif 

	free (df) ;	
	free (Q) ;
	free (t1) ;		
	free (a) ;
	free (f) ;
	free (lambda) ;
	return error ;
}

int Pref_MAP_Training (Pref_Settings * settings) 
{
	double error ;
	int index = 0 ;
	double * t1 ;
	double * t2 ;
	unsigned int i, j, u=0, v=0 ;
	double temp ;
	
	if (NULL == settings)
		return 0 ;

	// may refresh ALPHAS here 
	if (NULL != settings->alpha)
		Clear_Alphas(settings->alpha, settings->pairs) ;
	settings->alpha = Create_Alphas (settings) ;
	if (NULL == settings->alpha)
	{
		printf("fail to create ALPHAs.\n") ;
		exit(1) ;
	}

#ifdef _ORDINAL_DEBUG
	printf("Current Settings: \n") ;
	printf("Noise Variance: %.3f\n",settings->noisevar) ;
	printf("Kappa_O: %.3f\n",settings->kappa_o) ;
	if (0==settings->ardon)		
		printf("Kernel: %.3f\n",settings->kappa_a) ;
	else
	{
		printf("Kernel:\n") ;
		for (i=0;i<settings->pairs->dimen;i++)
			printf("%u ---  %.3f:\n", i, settings->kappa[i]) ;
	}
#endif

	// intialize function
	for (i=0;i<settings->pairs->count;i++)
		(settings->alpha+i)->pair->postmean = 0 ;
	// initialize the inverse matrix 
	t1 = (double *)malloc(settings->pairs->train*sizeof(double)) ;
	t2 = (double *)malloc(settings->pairs->train*sizeof(double)) ;
	if (NULL != settings->invcov) 
		free (settings->invcov) ;
	settings->invcov = (double *)malloc(settings->pairs->train*settings->pairs->train*sizeof(double)) ; // Sigma
	for (i=0;i<settings->pairs->count;i++)
	{
		if ((settings->alpha+i)->pair->fold < 0)
		{
			assert((settings->alpha+i)->pair->serial>0) ;
			v=0 ;
			for (j=0;j<i;j++)
			{
				if ((settings->alpha+j)->pair->fold < 0)
				{
					settings->invcov[u*settings->pairs->train+v] = 
						Calculate_Covfun(settings->alpha+i,settings->alpha+j,settings) ;
					settings->invcov[v*settings->pairs->train+u] = 
						settings->invcov[u*settings->pairs->train+v] ;
					v+=1 ;
				}
			}
			settings->invcov[u*settings->pairs->train+u] = Calculate_Covfun(settings->alpha+i,settings->alpha+i,settings) ;
			u+=1 ;
		}
	}
	assert(u==settings->pairs->train) ;	
	if (0 == cholesky(settings->invcov,settings->pairs->train,&temp))
		printf("Fatal Error: Sigma+1/Pi is not positive definite.\n") ;
	if (0 == inverse_from_cholesky(settings->invcov, t1, t2, settings->pairs->train) )		
		printf("Fatal Error: Sigma+1/Pi is not positive definite.\n") ;
	free(t1) ;
	free(t2) ;

	do
	{
		error = Pref_MAP_Update(settings) ; // clear sign
		index += 1 ;
#ifdef _ORDINAL_DEBUG
		printf("Loop %d --- %f\n", index, error) ;
#endif
	}
	while (error>TOL*TOL&&index<100) ;

	if (1 == Compute_MAP_Weights (settings))
		return 1 ;

#ifdef _ORDINAL_DEBUG
	/*for (index=0;index<(int)settings->pairs->count;index++)
		printf("%u --- post m %.3f - weight %.3f -alpha %.3f \n", 
		(settings->alpha+index)->pair->target, 
		(settings->alpha+index)->pair->postmean,
		(settings->alpha+index)->pair->weight,
		(settings->alpha+index)->alpha) ;*/
#endif
	return 0 ;
}



