#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "pref.h"
#include "neal_matrix.h"

// matrix product with matrix, Cmk = Amn * Bnk 
void brmul(double a[],double b[],int m,int n,int k,double c[]) 
{
	int i,j,l,u;
    for (i=0; i<=m-1; i++)
    for (j=0; j<=k-1; j++)
      { u=i*k+j; c[u]=0.0;
        for (l=0; l<=n-1; l++)
          c[u]=c[u]+a[i*n+l]*b[l*k+j];
      }
    return;
}

int Pref_LAPLACE_Evaluate_FuncGrad ( void * pointer, bfgs_FuncGrad * bfgs ) 
{
	Pref_Settings * settings ;
	Pref_Node * pair ;
	Alphas * alpha ;
	Alphas * alphab ;
	double * w ;
	double * ph ;

	double * lambda ;
	double * dsigma ;
	double * dmat ;
	double * df ;
	double * dfk ;
	double * dls ;
	double * ssmat ;
	double * sdmat ;

	double * zk ; // length = m
	double * gk ; // length = m, first order	
	double * hk ; // length = m, second order	
	double * tk ; // length = m	

	unsigned int u, v ;
	unsigned int i, j ;
	double phi, n, sk ;

	double * t1 ;
	double * t2 ;
	double temp ;
	double cbe ;
	unsigned int ii ; // settings->kappa	
	double sigma ;

	if (NULL == pointer || NULL == bfgs)
		return 1 ;
	settings = pointer ;

	Bfgs_Pref_Settings( settings,settings->bfgs ) ;

	// call EP routine to find posterior distribution 
	if (1 == Pref_MAP_Training (settings) )
	{
		printf("MAP failed to get equilibrium.\n") ;
		return 1 ;
	}

	// evaluate functional and gradient
	if (NULL != settings->invcov)
		free(settings->invcov) ;

	settings->invcov = (double*)malloc(settings->pairs->train*settings->pairs->train*sizeof(double)) ;
	dsigma = (double*)malloc(settings->pairs->train*settings->pairs->train*sizeof(double)) ;
	lambda = (double*)calloc(settings->pairs->train*settings->pairs->train,sizeof(double)) ;
	t1 = (double*)malloc(settings->pairs->train*sizeof(double)) ;
	t2 = (double*)malloc(settings->pairs->train*sizeof(double)) ;
	
	dmat = (double*)malloc(settings->pairs->train*settings->pairs->train*sizeof(double)) ;
	ssmat = (double*)malloc(settings->pairs->train*settings->pairs->train*sizeof(double)) ;
	sdmat = (double*)malloc(settings->pairs->train*settings->pairs->train*sizeof(double)) ;

	w = (double*)malloc(settings->pairs->train*sizeof(double)) ;	
	ph = (double*)malloc(settings->pairs->train*sizeof(double)) ;

	zk = (double*)malloc(settings->pairs->trainpair.count*sizeof(double)) ;		
	gk = (double*)malloc(settings->pairs->trainpair.count*sizeof(double)) ;		
	hk = (double*)malloc(settings->pairs->trainpair.count*sizeof(double)) ;		
	tk = (double*)malloc(settings->pairs->trainpair.count*sizeof(double)) ;			
	
	df = (double*)malloc(settings->pairs->train*sizeof(double)) ;			
	dfk = (double*)malloc(settings->pairs->train*sizeof(double)) ;			
	dls = (double*)calloc(settings->pairs->train*settings->pairs->train,sizeof(double)) ;

	sigma = sqrt(2.0*settings->noisevar) ;

	bfgs->fx = 0 ;
	for (i=0;i<settings->number;i++)
		bfgs->gx[i] = 0 ;

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
			w[u] = (settings->alpha+i)->pair->weight ; // alphas 
			ph[u] = 0 ;
			settings->invcov[u*settings->pairs->train+u] = Calculate_Covfun(settings->alpha+i,settings->alpha+i,settings) ;
			u+=1 ;
		}
	}
	assert(u==settings->pairs->train) ;
	
	// fx = 0.5*w*Sigma*w
	for (i=0;i<settings->pairs->train;i++)
	{
		for (j=0;j<i;j++)
			bfgs->fx += settings->invcov[i*settings->pairs->train+j]*w[i]*w[j] ;
		bfgs->fx += 0.5 * settings->invcov[i*settings->pairs->train+i]*w[i]*w[i] ;
	}

	// traverse training samples
	pair = settings->pairs->trainpair.front ;
	i=0;
	while (NULL != pair)
	{
		u = (settings->alpha+(pair->u-1))->pair->serial - 1 ;
		v = (settings->alpha+(pair->v-1))->pair->serial - 1 ;
		assert(u<pair->u&&u<settings->pairs->train) ;
		assert(v<pair->v&&v<settings->pairs->train) ;

		zk[i] = (settings->alpha+(pair->u-1))->pair->postmean 
			- (settings->alpha+(pair->v-1))->pair->postmean ; // f(u_k)-f(v_k)
		zk[i] = zk[i] / sigma ;
		n = normpdf(zk[i]) ;
		phi = 0.5 + normcdf(0,zk[i]) ;
		gk[i] = - n/phi/sigma ;
		hk[i] = ((n*n)/(phi*phi)+zk[i]*n/phi)/sigma/sigma ;
		tk[i] = (n/phi-2.0*(n*n*n)/(phi*phi*phi)-3.0*zk[i]*(n*n)/(phi*phi)-zk[i]*zk[i]*n/phi)/sigma/sigma/sigma ;

		sk = (zk[i]*zk[i]*n/phi+zk[i]*n*n/phi*phi-n/phi)/sigma/sigma*sqrt(2.0) ;
		ph[u] += sk ; 
		ph[v] -= sk ;

		bfgs->fx -= log(phi) ;		
		bfgs->gx[0] += zk[i]*n/phi/sigma ;

		// labmda	matrix
		lambda[(u)*settings->pairs->train+(u)] += hk[i] ;
		lambda[(v)*settings->pairs->train+(v)] += hk[i] ;
		lambda[(u)*settings->pairs->train+(v)] -= hk[i] ;
		lambda[(v)*settings->pairs->train+(u)] -= hk[i] ;
		
		i+=1 ;
		pair = pair->next ;
	}
	assert(i==settings->pairs->trainpair.count) ;

	if (0 == cholesky(settings->invcov,settings->pairs->train,&temp))
		printf("Fatal Error: Sigma+1/Pi is not positive definite.\n") ;

	for (i=0;i<settings->pairs->train;i++)
		bfgs->fx += log(settings->invcov[i*settings->pairs->train+i]) ;

	if (0 == inverse_from_cholesky(settings->invcov, t1, t2, settings->pairs->train) )		
		printf("Fatal Error: Sigma+1/Pi is not positive definite.\n") ;

	// Sigma^-1 + Lambda
	for (i=0;i<settings->pairs->train;i++)
	{
		for (j=0;j<i;j++)
		{
			dmat[i*settings->pairs->train+j] = settings->invcov[i*settings->pairs->train+j] + lambda[i*settings->pairs->train+j] ; 
			dmat[j*settings->pairs->train+i] = dmat[i*settings->pairs->train+j]; 
		}
		dmat[i*settings->pairs->train+i] = settings->invcov[i*settings->pairs->train+i] + lambda[i*settings->pairs->train+i] ; 
	}

	if (0 == cholesky(dmat,settings->pairs->train,&temp))
		printf("Fatal Error: Sigma+1/Pi is not positive definite.\n") ;
	
	for (i=0;i<settings->pairs->train;i++)
		bfgs->fx += log(dmat[i*settings->pairs->train+i]) ;
	// end of fx
	if (0 == inverse_from_cholesky(dmat, t1, t2, settings->pairs->train) )		
		printf("Fatal Error: Sigma+1/Pi is not positive definite.\n") ;

	// for gx[0] --- ln\sigma
	// df/ds
	for (i=0;i<settings->pairs->train;i++)	
	{
		df[i] = 0 ;
		for (j=0;j<settings->pairs->train;j++)
		{
			df[i]+=dmat[i*settings->pairs->train+j]*ph[j] ;
		}
	}

	// d lambda
	// traverse training samples
	for (i=0;i<settings->pairs->train;i++)
	{
		for (j=0;j<settings->pairs->train;j++)
		{
			dls[i*settings->pairs->train+j] = -2.0*lambda[i*settings->pairs->train+j]/sigma ;
		}
	}
	pair = settings->pairs->trainpair.front ;
	i=0;
	while (NULL != pair)
	{
		u = (settings->alpha+(pair->u-1))->pair->serial - 1 ;
		v = (settings->alpha+(pair->v-1))->pair->serial - 1 ;
		assert(u<pair->u&&u<settings->pairs->train) ;
		assert(v<pair->v&&v<settings->pairs->train) ;	
		// d labmda/d sigma		matrix
		dls[u*settings->pairs->train+v] += zk[i]*sqrt(2)*tk[i] ;
		dls[v*settings->pairs->train+u] += zk[i]*sqrt(2)*tk[i] ;
		dls[u*settings->pairs->train+u] -= zk[i]*sqrt(2)*tk[i] ;
		dls[v*settings->pairs->train+v] -= zk[i]*sqrt(2)*tk[i] ;
		// df
		dls[u*settings->pairs->train+v] += tk[i]*(-df[u]+df[v]) ;
		dls[v*settings->pairs->train+u] += tk[i]*(-df[u]+df[v]) ;
		dls[u*settings->pairs->train+u] += tk[i]*(df[u]-df[v]) ;
		dls[v*settings->pairs->train+v] += tk[i]*(df[u]-df[v]) ;
		i+=1 ;
		pair = pair->next ;
	}
	for (i=0;i<settings->pairs->train;i++)
	{
		for (j=0;j<settings->pairs->train;j++)
		{
			bfgs->gx[0] += 0.5*dmat[i*settings->pairs->train+j]*dls[j*settings->pairs->train+i] ;
		}
	}
	bfgs->gx[0] = bfgs->gx[0]*(-0.5)*sigma*sigma*sigma ; // d /d 1/sigma^2
	if (settings->kernel==LINEAR&&settings->ardon==1)
		bfgs->gx[0] = 0 ;

	// d ln kappa
	// for kernel parameters
	// calculate for each settings->kappa Gaussian parameter
	if (1 == settings->ardon)
	{
		for (ii=0;ii<settings->pairs->dimen;ii++)
		{			
			// calculate dsigmam for each settings->kappa parameter
			u=0 ;
			for (i=0;i<settings->pairs->count;i++)
			{
				alpha = settings->alpha + i ;	
				if ( alpha->pair->fold < 0 )
				{
					v=0 ;
					for (j=0;j<=i;j++)
					{
						alphab = settings->alpha + j ;
						if ( alphab->pair->fold < 0 )
						{							
							if ( GAUSSIAN == settings->kernel )
							{
								dsigma[u*settings->pairs->train+v]=(alpha->kernels[j]-settings->kappa_m)*(-(alpha->pair->
									point[ii]-alphab->pair->point[ii])*(alpha->pair->point[ii]-alphab->pair->point[ii])/2.0/settings->pairs->dimen) ;
							}
							else if ( LINEAR == settings->kernel )
							{
								dsigma[u*settings->pairs->train+v]=(alpha->pair->point[ii]*alphab->pair->point[ii])*settings->kappa_o ;
							}
							else if ( USERDEFINED == settings->kernel )
							{
								cbe = 0 ; // temp storage
								if (0==settings->pairs->featuretype[ii])
									cbe-=(alpha->pair->point[ii]-alphab->pair->point[ii]) 
									*(alpha->pair->point[ii]-alphab->pair->point[ii])/2.0 ;
								else if (alpha->pair->point[ii]!=alphab->pair->point[ii])
									cbe-=1.0/2.0 ;
								dsigma[u*settings->pairs->train+v] = (alpha->kernels[j]-settings->kappa_m) * cbe ;
							}
							else
								printf ("Error in kernel type.\n") ;
							dsigma[v*settings->pairs->train+u] = dsigma[u*settings->pairs->train+v] ;
							v += 1 ;
						}											
					}
					u += 1 ;
				}
			}
			brmul(dmat, settings->invcov, settings->pairs->train, settings->pairs->train, settings->pairs->train, ssmat) ;// VC*VC*a'*dsigma*a/2
			brmul(ssmat, dsigma, settings->pairs->train, settings->pairs->train, settings->pairs->train, sdmat) ;
			// VC*VC*a'*dsigma*a/2
			for (i=0;i<settings->pairs->train;i++)
			{						
				for (j=i+1;j<settings->pairs->train;j++)
					bfgs->gx[ii+settings->pairs->classes] -= w[i]*w[j]*dsigma[i*settings->pairs->train+j] ;		
				bfgs->gx[ii+settings->pairs->classes] -= 0.5*w[i]*w[i]*dsigma[i*settings->pairs->train+i] ;				
			}
			for (i=0;i<settings->pairs->train;i++)
				for (j=0;j<settings->pairs->train;j++)
					bfgs->gx[ii+settings->pairs->classes] += 0.5*sdmat[i*settings->pairs->train+j]*lambda[j*settings->pairs->train+i] ;
			//dfk
			for (i=0;i<settings->pairs->train;i++)
			{
				dfk[i] = 0 ;
				for (j=0;j<settings->pairs->train;j++)
					dfk[i] += sdmat[i*settings->pairs->train+j]*w[j] ; 
			}
			for (i=0;i<settings->pairs->train;i++)
			{
				for (j=0;j<settings->pairs->train;j++)
				{
					dls[i*settings->pairs->train+j] = 0 ;
				}
			}
			pair = settings->pairs->trainpair.front ;
			i=0;
			while (NULL != pair)
			{
				u = (settings->alpha+(pair->u-1))->pair->serial - 1 ;
				v = (settings->alpha+(pair->v-1))->pair->serial - 1 ;
				assert(u<=pair->u&&u<=settings->pairs->train) ;
				assert(v<=pair->v&&v<=settings->pairs->train) ;	
				// df
		dls[u*settings->pairs->train+v] += tk[i]*(-dfk[u]+dfk[v]) ;
		dls[v*settings->pairs->train+u] += tk[i]*(-dfk[u]+dfk[v]) ;
		dls[u*settings->pairs->train+u] += tk[i]*(dfk[u]-dfk[v]) ;
		dls[v*settings->pairs->train+v] += tk[i]*(dfk[u]-dfk[v]) ;
				i+=1 ;
				pair = pair->next ;
			}
			for (i=0;i<settings->pairs->train;i++)
				for (j=0;j<settings->pairs->train;j++)
					bfgs->gx[ii+settings->pairs->classes] += 0.5*dmat[i*settings->pairs->train+j]*dls[j*settings->pairs->train+i] ;

bfgs->gx[ii+settings->pairs->classes] += settings->regular*settings->kappa[ii] ;
bfgs->fx += 0.5*settings->regular*settings->kappa[ii]*settings->kappa[ii] ;
//bfgs->gx[ii+settings->pairs->classes] += settings->regular ;
//bfgs->fx += settings->regular*settings->kappa[ii] ;

			bfgs->gx[ii+settings->pairs->classes] = bfgs->gx[ii+settings->pairs->classes] * settings->kappa[ii] ;
		}
	}
	else
	{
		u=0 ;
		for (i=0;i<settings->pairs->count;i++)
		{
			alpha = settings->alpha + i ;
			if ( alpha->pair->fold < 0 )
			{
				v = 0 ;
				for (j=0;j<=i;j++)
				{
					alphab = settings->alpha + j ;
					if ( alphab->pair->fold < 0 )
					{							
						if ( GAUSSIAN == settings->kernel )
						{
							cbe = 0 ; // temp storage
							for (ii=0;ii<settings->pairs->dimen;ii++)
								cbe-=(alpha->pair->point[ii]-alphab->pair->point[ii])
								*(alpha->pair->point[ii]-alphab->pair->point[ii]) ;
							dsigma[u*settings->pairs->train+v] = (alpha->kernels[j]-settings->kappa_m)*cbe/2.0/settings->pairs->dimen ;
						} 
						else if (LINEAR == settings->kernel )
						{
							cbe = 0 ; // temp storage
							for (ii=0;ii<settings->pairs->dimen;ii++)
								cbe += alpha->pair->point[ii]*alphab->pair->point[ii] ;
							dsigma[u*settings->pairs->train+v] = cbe*settings->kappa_o ;
						}
						else if ( USERDEFINED == settings->kernel )
						{
							cbe = 0 ; // temp storage
							for (ii=0;ii<settings->pairs->dimen;ii++)
							{
								if (0==settings->pairs->featuretype[ii])
									cbe-=(alpha->pair->point[ii]-alphab->pair->point[ii])
									*(alpha->pair->point[ii]-alphab->pair->point[ii]) ;
								else if (alpha->pair->point[ii]!=alphab->pair->point[ii])
									cbe-=1.0 ;
							}
							dsigma[u*settings->pairs->train+v] = (alpha->kernels[j]-settings->kappa_m) * cbe / 2.0 ;
						}
						else
							printf ("Error in kernel type.\n") ;
						dsigma[v*settings->pairs->train+u] = dsigma[u*settings->pairs->train+v] ;
						v += 1 ;
					}
				}
				u += 1 ;
			}
		}
			brmul(dmat, settings->invcov, settings->pairs->train, settings->pairs->train, settings->pairs->train, ssmat) ;// VC*VC*a'*dsigma*a/2
			brmul(ssmat, dsigma, settings->pairs->train, settings->pairs->train, settings->pairs->train, sdmat) ;
			// VC*VC*a'*dsigma*a/2
			for (i=0;i<settings->pairs->train;i++)
			{						
				for (j=i+1;j<settings->pairs->train;j++)
					bfgs->gx[settings->pairs->classes] -= w[i]*w[j]*dsigma[i*settings->pairs->train+j] ;		
				bfgs->gx[settings->pairs->classes] -= 0.5*w[i]*w[i]*dsigma[i*settings->pairs->train+i] ;				
			}
			for (i=0;i<settings->pairs->train;i++)
				for (j=0;j<settings->pairs->train;j++)
					bfgs->gx[settings->pairs->classes] += 0.5*sdmat[i*settings->pairs->train+j]*lambda[j*settings->pairs->train+i] ;
			//dfk
			for (i=0;i<settings->pairs->train;i++)
			{
				dfk[i] = 0 ;
				for (j=0;j<settings->pairs->train;j++)
					dfk[i] += sdmat[i*settings->pairs->train+j]*w[j] ;
				//printf("dfk=%f\n",dfk[i]) ;
			}
			for (i=0;i<settings->pairs->train;i++)
			{
				for (j=0;j<settings->pairs->train;j++)
				{
					dls[i*settings->pairs->train+j] = 0 ;
				}
			}
			pair = settings->pairs->trainpair.front ;
			i=0;
			while (NULL != pair)
			{
				u = (settings->alpha+(pair->u-1))->pair->serial - 1 ;
				v = (settings->alpha+(pair->v-1))->pair->serial - 1 ;
				assert(u<=pair->u&&u<=settings->pairs->train) ;
				assert(v<=pair->v&&v<=settings->pairs->train) ;	
				// df
		dls[u*settings->pairs->train+v] += tk[i]*(-dfk[u]+dfk[v]) ;
		dls[v*settings->pairs->train+u] += tk[i]*(-dfk[u]+dfk[v]) ;
		dls[u*settings->pairs->train+u] += tk[i]*(dfk[u]-dfk[v]) ;
		dls[v*settings->pairs->train+v] += tk[i]*(dfk[u]-dfk[v]) ;
				i+=1 ;
				pair = pair->next ;
			}
			for (i=0;i<settings->pairs->train;i++)
			{
				for (j=0;j<settings->pairs->train;j++)
					bfgs->gx[settings->pairs->classes] += 0.5*dmat[i*settings->pairs->train+j]
													*dls[j*settings->pairs->train+i] ;
			}

		bfgs->gx[settings->pairs->classes] = bfgs->gx[settings->pairs->classes] * settings->kappa_a ;

	if (settings->kernel==LINEAR)
		bfgs->gx[settings->pairs->classes] = 0 ;
	}
	

 //       for (i=0;i<settings->number;i++)
 //       {
		//if (i<settings->pairs->classes)
		//	bfgs->gx[i]=0;
 //       }
//bfgs->gx[0] = 0 ;
	
#ifdef _ORDINAL_DEBUG
	//if (bfgs->x[0]>=100)
	//bfgs->gx[settings->number-1] = 0 ;
	printf("functional %f \n", bfgs->fx) ;
	for (i=0;i<settings->number;i++)
	{
		printf("gardient %f --- x %f\n", bfgs->gx[i], bfgs->x[i]) ;
	}
#endif 
	free(dls) ;
	free(dfk) ;
	free(df) ;
	free(tk) ;
	free(hk) ;
	free(gk) ;
	free(zk) ;
	free(ph) ;
	free(w) ;
	free(ssmat) ;
	free(sdmat) ;	
	free(dmat) ;
	free(t2) ;
	free(t1) ;
	free(lambda) ;
	free(dsigma) ;
	return 0 ;
}

int	Pref_LAPLACE_Training (Pref_Settings * settings) 
{
	unsigned int i ;

	if (NULL == settings)
		return 1 ;

	printf("Laplacian Approximation around MAP estimate ...\n") ;
	if ( 1 == lbfgsb_minfunc ( Pref_LAPLACE_Evaluate_FuncGrad, settings, settings->bfgs ) )
	{
		printf("fail to find local minimum.\n") ;
		return 1;
	}
	// save best funcgrad to settings 
	Bfgs_Pref_Settings( settings, settings->bfgs ) ;
	printf("Optimal Settings: \n") ;
	printf("Noise Variance: %f\n",settings->noisevar) ;
	printf("Kappa_O: %f\n",settings->kappa_o) ;
	if (0==settings->ardon)		
		printf("Kernel: %f\n",settings->kappa_a) ;
	else if (settings->pairs->dimen<10) 
	{
		printf("Kernel:\n") ;
		for (i=0;i<settings->pairs->dimen;i++)
			printf("%u ---  %f:\n", i, settings->kappa[i]) ;
	}
	tstart();
	Pref_MAP_Training (settings) ;
	tend();
	settings->time = tval() ;
	return 0 ;
}

