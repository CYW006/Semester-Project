#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include "pref.h"
#include "neal_matrix.h"

extern double lerrf(double) ;
extern double fromb(double mean, double sig, double eps) ;

int Pref_Prediction (Pref_Settings * settings) 
{
	Pref_Node * pair ;
	Data_Node * node ;
	double * t1 ; 
	double * t2 ;
	double * lambda ;
	double * kcvx ;
	double st ;
	double zk, hk, n, phi ;
	double temp, sigma ;
	unsigned int i, j, k, u, v ;
	char buf[2048] ;
	FILE * fid ;
	FILE * fid_guess ; 
	unsigned int error = 0, abserr = 0 ;

	assert(NULL != settings) ;

	// compute the invcov
		if (NULL != settings->invcov)
		free(settings->invcov) ;

	settings->invcov = (double*)malloc(settings->pairs->train*settings->pairs->train*sizeof(double)) ;
	lambda = (double*)calloc(settings->pairs->train*settings->pairs->train,sizeof(double)) ;
	t1 = (double*)malloc(settings->pairs->train*sizeof(double)) ;
	t2 = (double*)malloc(settings->pairs->train*sizeof(double)) ;	

	// initial Sigma
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
					assert((settings->alpha+j)->pair->serial>0) ;
					settings->invcov[u*settings->pairs->train+v] = 
						Calculate_Covfun(settings->alpha+i,settings->alpha+j,settings) ;
					settings->invcov[v*settings->pairs->train+u] = 
						settings->invcov[u*settings->pairs->train+v] ;
					v+=1 ;
				}
			}
			settings->invcov[u*settings->pairs->train+u] = 
				Calculate_Covfun(settings->alpha+i,settings->alpha+i,settings)-DEF_JITTER ;
			u+=1 ;
		}
	}
	assert(u==settings->pairs->train) ;
	// traverse training samples
	sigma =  sqrt(2.0*settings->noisevar) ;
	// traverse training samples
	pair = settings->pairs->trainpair.front ;
	i=0;
	while (NULL != pair)
	{
		u = (settings->alpha+(pair->u-1))->pair->serial - 1 ;
		v = (settings->alpha+(pair->v-1))->pair->serial - 1 ;
		assert(u<pair->u&&u<settings->pairs->train) ;
		assert(v<pair->v&&v<settings->pairs->train) ;

		zk = (settings->alpha+(pair->u-1))->pair->postmean 
			- (settings->alpha+(pair->v-1))->pair->postmean ; // f(u_k)-f(v_k)
		zk = zk / sigma ;
		n = normpdf(zk) ;
		phi = 0.5 + normcdf(0,zk) ;
		hk = ((n*n)/(phi*phi)+zk*n/phi)/sigma/sigma ;
		// labmda	matrix
		lambda[(u)*settings->pairs->train+(u)] += hk ;
		lambda[(v)*settings->pairs->train+(v)] += hk ;
		lambda[(u)*settings->pairs->train+(v)] -= hk ;
		lambda[(v)*settings->pairs->train+(u)] -= hk ;		
		i+=1 ;
		pair = pair->next ;
	}
	assert(i==settings->pairs->trainpair.count) ;
	for (i=0;i<settings->pairs->train;i++)
	{
		lambda[i*settings->pairs->train+i] += DEF_JITTER ;
		for (j=0;j<settings->pairs->train;j++)
			assert(lambda[i*settings->pairs->train+j]==lambda[j*settings->pairs->train+i]) ;
	}
	if (0 == cholesky(lambda,settings->pairs->train,&temp))
		printf("1Fatal Error: Sigma+1/Pi is not positive definite.\n") ;
	if (0 == inverse_from_cholesky(lambda, t1, t2, settings->pairs->train) )		
		printf("2Fatal Error: Sigma+1/Pi is not positive definite.\n") ;

	for (i=0;i<settings->pairs->train;i++)
	{
		for (j=0;j<i;j++)
		{
			settings->invcov[i*settings->pairs->train+j] += lambda[i*settings->pairs->train+j] ;
			settings->invcov[j*settings->pairs->train+i] = settings->invcov[i*settings->pairs->train+j] ;
		}
		settings->invcov[i*settings->pairs->train+i] += lambda[i*settings->pairs->train+i] ;
	}

	if (0 == cholesky(settings->invcov,settings->pairs->train,&temp))
		printf("3Fatal Error: Sigma+1/Pi is not positive definite.\n") ;
	if (0 == inverse_from_cholesky(settings->invcov, t1, t2, settings->pairs->train) )		
		printf("4Fatal Error: Sigma+1/Pi is not positive definite.\n") ;

	// compute predictions
	node = settings->pairs->front ;
	while (NULL != node)
	{
		// for each point
		node->pred_func = 0 ;
		// cache kcx 
		kcvx = (double*)malloc(settings->pairs->train*sizeof(double)) ;
		u=0 ;
		for (i=0;i<settings->pairs->count;i++)
		{
			if ((settings->alpha+i)->pair->fold<0)
			{
				kcvx[u] = Calc_Covfun((settings->alpha+i)->pair->point,node->point,settings) ;
				node->pred_func += (settings->alpha+i)->pair->weight*kcvx[u];
				u += 1 ;
			}
			else
				assert((settings->alpha+i)->pair->weight==0) ;
		}
		assert(u==settings->pairs->train) ;
		node->pred_var = Calc_Covfun(node->point,node->point,settings) ; ;
		// compute sigmax
		for (i=0;i<settings->pairs->train;i++)
		{
			for (k=0;k<i;k++)
				node->pred_var -= 2.0*kcvx[i]*kcvx[k]*settings->invcov[i*settings->pairs->train+k] ;
			node->pred_var -= kcvx[i]*kcvx[i]*settings->invcov[i*settings->pairs->train+i] ;
		}
		//assert(node->pred_var >= 0) ;
		if (node->pred_var <= 0)
			printf("warning: negative variance in predictive distribution %f.\n", node->pred_var) ;
		free(kcvx) ;
		node=node->next ;
	}

	// save the function values
	sprintf(buf,"%s.func",settings->trainfile) ;
	fid = fopen(buf,"w+t") ;
	if (NULL != fid)
	{
		node = settings->pairs->front ;
		while (NULL != node)
		{
			// for each new 
			fprintf(fid, "%f\n",node->pred_func) ;
			node=node->next ;
		}
		fclose(fid) ;
	}
	
	// save the variance values
	sprintf(buf,"%s.conf",settings->trainfile) ;
	fid = fopen(buf,"w+t") ;
	if (NULL != fid)
	{
		node = settings->pairs->front ;
		while (NULL != node)
		{
			// for each new 
			fprintf(fid, "%f\n", node->pred_var) ;
			node=node->next ;
		}
		fclose(fid) ;
	}
	
	// loading test data 
	Create_Pref_List (&(settings->pairs->testpair)) ;
	Pref_Loadpair (&(settings->pairs->testpair), settings->testfile) ;

	// compute predictions
	sprintf(buf,"%s.prob",settings->testfile) ;
	fid = fopen(buf,"w+t") ;
	sprintf(buf,"%s.guess",settings->testfile) ;
	fid_guess = fopen(buf,"w+t") ;
	pair = settings->pairs->testpair.front ;	
	while (NULL != pair)
	{
		// compute predictive probability
		zk = (settings->alpha+(pair->u-1))->pair->pred_func 
			- (settings->alpha+(pair->v-1))->pair->pred_func ;
		hk = 2.0*settings->noisevar
			+(settings->alpha+(pair->u-1))->pair->pred_var
			+(settings->alpha+(pair->v-1))->pair->pred_var ;
		
		// covariance
		st = 0 ;
		/*st = Calc_Covfun((settings->alpha+(pair->u-1))->pair->point,(settings->alpha+(pair->v-1))->pair->point,settings)  ;
		u=0 ;
		for (i=0;i<settings->pairs->count;i++)
		{
			if ((settings->alpha+i)->pair->fold<0)
			{
				v = 0 ;
				for (j=0;j<settings->pairs->count;j++)
				{
					if ((settings->alpha+j)->pair->fold<0)
					{
						st -= settings->invcov[u*settings->pairs->train+v]
							*Calc_Covfun((settings->alpha+i)->pair->point,(settings->alpha+(pair->u-1))->pair->point,settings)
							*Calc_Covfun((settings->alpha+j)->pair->point,(settings->alpha+(pair->v-1))->pair->point,settings) ;
						v+=1 ;
					}
					else
						assert((settings->alpha+j)->pair->weight==0) ;
				}
				u+=1 ;
			}
		}*/
		// the covriance
		assert((hk-2.0*st)>0) ;
		zk = zk/sqrt(hk-2.0*st) ;
		phi = 0.5 + normcdf(0,zk) ;
		if (zk<0)
		{
			error += 1 ;
			abserr += 1 ;
		}
		if (NULL != fid)
			fprintf(fid,"%f\n",phi) ;
		if (NULL != fid_guess)
		{
			fprintf(fid_guess,"%d %f %d\n",phi>0.5?+1:-1,phi,1) ;
			fprintf(fid_guess,"%d %f %d\n",phi<0.5?+1:-1,1.0-phi,-1) ;
		}
		pair=pair->next ;
	}
	if (NULL != fid)
		fclose(fid) ;
	if (NULL != fid_guess)
		fclose(fid_guess) ;

	if (settings->pairs->deviation > 0)
		printf("Testing error number %u with absolute error %u.\n", error, abserr) ;

	fid = fopen ("gpref_lap_batch.log","a+t") ;

	if (NULL != fid)
	{
		fprintf(fid,"%u %u %f %f\n", error, abserr, (double)abserr/(double)(settings->pairs->testpair.count), settings->time) ;
		fclose(fid) ;
	}
	// save log files
	free(t1) ;
	free(t2) ;
	return 0 ;
}

int Dumping_Pref_Settings (Pref_Settings * settings)
{
	FILE * fid ;
	char buf[2048] ;
	unsigned int i, j ;
	double temp ;

	if (NULL == settings)
		return 1 ;

	sprintf(buf,"%s.log",settings->pairs->filename) ;
	fid = fopen (buf,"w+t") ;
	if (NULL != fid)
	{
		// add testing information
		sprintf(buf, "final settings\n") ;
		fwrite (buf, sizeof(char), strchr(buf,'\0')-buf, fid) ;
		fprintf(fid,"kappa_o = %f\n",settings->kappa_o) ;
		if (0 == settings->ardon)
			fprintf(fid,"kappa = %f\n",settings->kappa_a) ;
		else 
		{
			for (i=0;i<settings->pairs->dimen;i++)
				fprintf(fid,"kappa%d = %f\n", i+1, settings->kappa[i]) ;				
		}
		fprintf(fid,"kappa_m = %f\n", settings->kappa_m) ;
		fprintf(fid,"noise variance = %f\n", settings->noisevar) ;
		for (i=0;i<settings->pairs->classes-1;i++)
			fprintf(fid,"threshold%d = %f\n", i+1, settings->thresholds[i]) ;
		fclose(fid) ;
	}
	if (1==settings->ardon)
	{
       		sprintf(buf,"%s.ard",settings->pairs->filename) ;
        	fid = fopen (buf,"w+t") ;
        	if (NULL != fid)
        	{
				// add testing information
				for (i=0;i<settings->pairs->dimen;i++)
					fprintf(fid,"%.10f\n", settings->kappa[i]) ;
			}
			fclose(fid) ;
	}
	if (LINEAR==settings->kernel)
	{
       		sprintf(buf,"%s.lw",settings->pairs->filename) ;
        	fid = fopen (buf,"w+t") ;
        	if (NULL != fid)
        	{
				// add testing information
				for (i=0;i<settings->pairs->dimen;i++)
				{
					temp = 0 ;
					for (j=0;j<settings->pairs->count;j++)
					{
						if ((settings->alpha+j)->pair->fold < 0)
						{
							temp += (settings->alpha+j)->pair->weight
								*(settings->alpha+j)->pair->point[i] ;
						}
						else
							assert((settings->alpha+j)->pair->weight==0) ;
					}
					fprintf(fid,"%.10f\n", temp) ;
				}
			}
			fclose(fid) ;
	}
	return 0 ;
}

// end of gpor_predict.cpp

