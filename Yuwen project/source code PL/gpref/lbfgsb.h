#ifdef  __cplusplus
extern "C" {
#endif

#ifndef _LBFGSB_H
#define _LBFGSB_H

typedef long int integer;
typedef unsigned long int uinteger;
typedef char *address;
typedef short int shortint;
typedef float real;
typedef double doublereal;
typedef long int logical;
typedef short int shortlogical;
typedef char logical1;
typedef char integer1;
typedef short flag;
typedef short ftnlen;
typedef short ftnint;

#define s_cmp(a,b,c,d) (strncmp ((a),(b), (c) >= (d) ? (d) : (c)))
#define s_copy(a,b,c,d) (strcpy((a),(b)))

typedef struct bfgssettings
{
	unsigned int number ;
	double * x ;
	double fx ;
	double * gx ;
	unsigned int iternum ;

} Bfgs_Settings ;

Bfgs_Settings * CreateBfgsSettings(unsigned int);
void ClearBfgsSettings(Bfgs_Settings *);
int lbfgsb_minfunc(int(* f)(void *,Bfgs_Settings *),void *,Bfgs_Settings *); 
int lbfgs_minfunc(int(* f)(void *,Bfgs_Settings *),void *,Bfgs_Settings *); 

#endif

#ifdef  __cplusplus
}
#endif
// end of lbfgsb.h

