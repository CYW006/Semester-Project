// header file --- ordinal.h 

// Gaussian Processes for Ordinal Regression

// Chu Wei (C) Copyright 2004


#ifdef  __cplusplus
extern "C" {
#endif

#ifndef _PREFERENCE_REGRESSION
#define _PREFERENCE_REGRESSION
	
#define MINNUM          (3)				// at least two
#define LENGTH          (307200)		// maximum value of line length in data file 
#define PI				(3.141592654)

//********CHOOSE ONE MODE ONLY**********//
#ifdef _GPREF_EP
#define _PREF_EP
#else
#define _PREF_LP
#endif
//**************************************//

//#define _PREF_DEBUG
//#define _GPREF_ARD

typedef enum _Data_Type
{
	PREFERENCE = 6 ,
	REGRESSION = 3 ,
	ORDINALREGRESSION = 5 ,
	MULTICLASS = 4 ,
	CLASSIFICATION = 1 ,
	UNKNOWN = 0 ,

} Data_Type ;

typedef enum _Kernel_Name
{
	GAUSSIAN = 0 ,	
	POLYNOMIAL = 1 ,
	USERDEFINED = 3 ,
	LINEAR = 4 ,

} Kernel_Name ;

typedef struct _Pref_Node 
{
	long unsigned int index ;
	unsigned long int u ; 
	unsigned long int v ;
	struct _Pref_Node * next ;      // point to next node in the list

} Pref_Node ;

typedef struct _Pref_List 
{
	unsigned long int count ;       // total number of pairs 
	Pref_Node * front ;             // point to first node in the list
	Pref_Node * rear ;              // point to last node in the list

} Pref_List ;

typedef struct _Data_Node 
{
	long unsigned int index ;       // line number in data file loaded 
	double * point ;                // point to one input point
	int fold ;						// fold label 
	int serial ;
	unsigned int target ;			// class label	
	unsigned int pred_label ;
	double pred_func ;
	double pred_prob ;              // predictive probability 
	double pred_var ;
	
	struct _Data_Node * prev ;      // point to next node in the list
	struct _Data_Node * next ;      // point to next node in the list

	double weight ;					// contribution for the predictor
	// Moment Matching
	double epmean ;					// individual mean in EP
	double epinvvar ;				// individual 1/variance in EP
	double epamp ;					// individual amplitude in EP
	double postmean ;				// mean in posterior distribution

} Data_Node ;

typedef struct _Data_List 
{
	Data_Type datatype ;            // regression problem or classification
	int normalized_input ;          // point data_node normalized or not 
	int normalized_output ;         // target data_node normalized or not 
	
	unsigned long int count ;       // total number of samples 
	unsigned long int dimen ;       // dimension of input vector
	unsigned int classes ;			// number of classes 

	int kfold ;						// for k-fold only

	int * featuretype ;				// for userdefined Data_Type
	
	int * labels ;	
	int * labelnum ;

	double * weights ;				// for linear models

	unsigned int train ;			// distinct samples
	Pref_List trainpair ;
	Pref_List testpair ;

	char * filename ;

	unsigned int i_ymax ;
	unsigned int i_ymin ;
	double mean ;                   // mean of output
	double deviation ;              // deviation of output
	double * x_mean ;               // mean of input
	double * x_devi ;               // standard deviation of input
	
	Data_Node * front ;             // point to first node in the list
	Data_Node * rear ;              // point to last node in the list

} Data_List ;

typedef struct _Alphas
{
	double alpha ;
	double beta ;
	double nu ;
	double loomean ;
	double loovar ;
	double z1 ;
	double z2 ;
	double hnew ; // new posterior mean
	double mnew ; // new individual mean
	double pnew ; // new individual variance
	double snew ; // new individual amplitude

	double * kernels ;
	double * postcov ;
	double * p_cache ;              // save Pi(x) for CG 
	
	Data_Node * pair ;              // point to the corresponding pair 

} Alphas ;

typedef struct _bfgs_FuncGrad
{
	unsigned int var_number ;
	double * x ;
	double fx ;
	double * gx ;	
	double norm2gx ;    // true norm square |g(x)|^2
	double maskednorm ; // norm |g(x)|^2 of masked gradient 
	unsigned int iternum ;

} bfgs_FuncGrad ;

typedef struct _Pref_Settings
{
	Kernel_Name kernel ;
	unsigned int number ;
	unsigned int p  ;	
	unsigned int kfoldcv  ;
	int ardon ;
	int cacheall ;
	double kappa_o ;	
	double noisevar ;				// noise variance
	double kappa_a ;
	double * kappa ;				// kappa for ARD variables (d)
	double kappa_m ;
	double * thresholds ;			// ordinal threshold (r-1)
	double * intervals ; 			// threshold interval (r-2)
	
	double regular ;
	double time ;

	char * trainfile ;              // the name of train data file 
	char * testfile ;               // the name of test data file 
	
	double * invcov ;
	
	bfgs_FuncGrad *	bfgs ;

	struct _Data_List * pairs ;     // this is a reference from def_Settings
	struct _Alphas * alpha ;        // Pointers to Alphas matrix 

} Pref_Settings ;

#define DEF_KFOLDCV			  (2)
#define DEF_NORMALIZEINPUT    (0)// 1 - YES TRUE ; 0 - NO FALSE
#define DEF_NORMALIZETARGET   (0)
#define DEF_CACHEALL		  (1)
#ifdef _GPREF_ARD
#define DEF_ARDON			  (1)
#else
#define DEF_ARDON 			  (0)
#endif
#define DEF_KAPPA			  (1)
#define DEF_NOISEVAR		  (1)
#define DEF_KERNEL			  (LINEAR)
#define DEF_KAPPA_M			  (0)
#define DEF_KAPPA_O			  (1)
#define DEF_P				  (2)
#define DEF_JITTER			  (0.001)
#define DEF_REGULAR			  (0.01)
#define EPS					  (0.000001)
#define TOL					  (0.001)



int Pref_Loadfile ( Data_List * pairs, char * inputfilename ) ;
int Pref_Loadpair ( Pref_List * pref, char * inputfilename ) ;

Pref_Settings * Create_Pref_Settings(Data_List * list) ;
int Clear_Pref_Settings( Pref_Settings * settings) ;

int Create_Pref_List(Pref_List * list) ;
int Clear_Pref_List(Pref_List * list) ;

bfgs_FuncGrad * Create_Bfgs_FuncGrad ( unsigned int number ) ;
int Pref_Bfgs_Settings ( Pref_Settings * settings, bfgs_FuncGrad * funcgrad ) ;
int Bfgs_Pref_Settings ( Pref_Settings * settings, bfgs_FuncGrad * funcgrad ) ;
void Clear_Bfgs_FuncGrad ( bfgs_FuncGrad * funcgrad ) ;

int lbfgsb_minfunc ( int (* Evaluate_FuncGrad)(void *, bfgs_FuncGrad *),	void * , bfgs_FuncGrad * ) ;

int Pref_Prediction (Pref_Settings * settings) ;
int Dumping_Pref_Settings (Pref_Settings * settings) ;


int Compute_EP_Weights (Pref_Settings * settings) ;

double lerrf(double x) ;
double fromb_t1(int index, Pref_Settings * settings) ;
double fromb_t2(int index, Pref_Settings * settings) ;
double fromb_t3(int index, Pref_Settings * settings) ;
double fromb_t4(int index, Pref_Settings * settings) ;
double fromb_t5(int index, Pref_Settings * settings) ;

//int Ordinal_EPEM_Evaluate_FuncGrad ( void * pointer, bfgs_FuncGrad * bfgs ) ;
//int	Ordinal_EPEM_Training (Pref_Settings * settings) ;

int	Pref_LAPLACE_Training (Pref_Settings * settings) ;
int Pref_MAP_Training (Pref_Settings * settings) ;

double normal(double x) ;
int Is_Data_Empty ( Data_List * list ) ;
int Create_Data_List ( Data_List * list ) ;
int Clear_Data_List ( Data_List * list ) ;
int Add_Data_List ( Data_List * list, Data_Node * node ) ;

Data_Node * Create_Data_Node ( long unsigned int index, double * point, int y ) ;


Alphas * Create_Alphas (Pref_Settings * list) ;
int Clear_Alphas (Alphas * alpha, Data_List * list) ;

double Calculate_Covfun( struct _Alphas * ai, struct _Alphas * aj, Pref_Settings * settings ) ;
double Calc_Covfun( double * pi, double * pj, Pref_Settings * settings ) ;

//int Ordinal_EP_Training (Pref_Settings *) ;

int Add_Label_Data_List ( Data_List * list, Data_Node * node ) ;
int Clear_Label_Data_List ( Data_List * list ) ;
//int Newton_Soft_Max (Data_List * list) ;

int Create_Kfold_List (Data_List * list, int KFOLDCV) ;
Data_Node ** Create_Fold_List (Data_List * list, Data_List * train, Data_List * test, int fold) ;
int Clear_Fold_List (Data_List * list, Data_List * train, Data_List * test, Data_Node **) ;


int Find_Out_Label (int * labels, int size, int inquiry) ;
int * Test_Soft_Max(Data_List * train, Data_List * test) ;
int Initial_Pcache (Alphas * alpha, Data_List * list) ;

double normpdf(double x) ;
double normcdf(double ca, double eb) ;
//timing routines
void tstart(void) ;
void tend(void) ;
double tval() ;

int conjugate_gradient (double * A, double * B, double * x, unsigned int n) ;

#endif

#ifdef  __cplusplus
}
#endif

