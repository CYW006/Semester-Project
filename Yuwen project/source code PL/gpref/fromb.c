#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pref.h"
// romberg for normcdf

double normpdf(double x)
{
	return exp(-0.5*x*x)/sqrt(2.0*PI) ;
}

double normcdf(double ca, double eb)
{ 
    int m,n,i,k;
	double eps ;
    double y[20],h,ep,p,x,s,q=0,a,b ;

	eps = EPS*EPS ;
	if (ca>eb)
	{	a=eb; b=ca;	}
	else
	{	a=ca; b=eb;	}

    h=b-a;
    y[0]=h*(normpdf(a)+normpdf(b))/2.0;
    m=1; n=1; ep=eps+1.0;
    while ((ep>=eps)&&(m<=19))
      { p=0.0;
        for (i=0;i<=n-1;i++)
          { x=a+(i+0.5)*h;
            p=p+normpdf(x);
          }
        p=(y[0]+h*p)/2.0;
        s=1.0;
        for (k=1;k<=m;k++)
          { s=4.0*s;
            q=(s*p-y[k-1])/(s-1.0);
            y[k-1]=p; p=q;
          }
        ep=fabs(q-y[m-1]);
        m=m+1; y[m-1]=q; n=n+n; h=h/2.0;
      }
	if (ca>eb)
    return(-q) ;
	else
    return(q) ;
}

double gaussian(double x, double mean, double sig)
{
	if (sig>0)
		return exp(-0.5*(x-mean)*(x-mean)/sig/sig)/sqrt(2.0*PI)/sig ;
	else
	{
		printf("negative variance in Gaussian.\n") ;
		return 0 ;
	}
}
//end of fromb.c

