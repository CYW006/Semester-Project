/*******************************************************************************\

	loadfile.c in Sequential Minimal Optimization ver2.0 
	
	loads data file from disk in data list.

	Chu Wei Copyright(C) National Univeristy of Singapore
	Create on Jan. 16 2000 at Control Lab of Mechanical Engineering 
	Update on Aug. 23 2001 

\*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <limits.h>
#include "pref.h"

/*******************************************************************************\

	BOOL Pref_Loadfile ( Pairs * pairs, char * filename, unsigned int inputdim ) 
	
	load data file settings->inputfile, and create the data list Pairs 
	input:  the pointers to pairs and filename
	output: 0 or 1

\*******************************************************************************/

int Pref_Loadfile ( Data_List * pairs, char * inputfilename ) 
{ 

	FILE * smo_stream ;
	FILE * smo_target = NULL ;
	char * pstr = NULL ;
	char buf[LENGTH] ;
	char * temp ;
	int dim = -1 ;
	unsigned long index = 1 ;
	unsigned int result ;
	int var = 0, chg = 0 ;
	double * point = NULL ;
	int y = 0 ;
	int i = 0, j = 0 ;
	int inputdim = 0 ;
	double mean = 0 ;
	double ymax = LONG_MIN ;
	double ymin = LONG_MAX ;
	double * xmean = NULL;
	Data_Node * node = NULL ;
	int t0=0 , tr=0 ;
	FILE * fid ;
	unsigned int sz ;

	Data_List label ;

	if ( NULL == pairs || NULL == inputfilename )
		return 1 ;
	
	Clear_Data_List( pairs ) ;
	Create_Data_List( &label ) ;

	if( (smo_stream = fopen( inputfilename, "r+t" )) == NULL )
	{
		//printf( "can not open the file %s.\n", inputfilename );
		return 1 ;
	}
	
	// save file name 
	var = strlen( inputfilename ) ;
	if (NULL != pairs->filename) 
		free(pairs->filename) ;
	pairs->filename = (char*)malloc((var+1)*sizeof(char)) ;
	if (NULL == pairs->filename)
	{
		printf("fail to malloc for pairs->filename.\n") ;
		exit(0) ;
	}
	strncpy(pairs->filename,inputfilename,var) ;
	pairs->filename[var]='\0' ;

	// check the input dimension here

	if ( NULL == fgets( buf, LENGTH, smo_stream ))
	{
		printf( "fgets error in reading the first line.\n" );
		fclose( smo_stream );
		return 1 ;
	}
	
	var = strlen( buf ) ;
	
	if (var >= LENGTH-1) 
	{
		printf( "the line is too long in the file %s.\n", inputfilename );
		fclose( smo_stream );
		return 1 ;
	}
	
	if (0 < var)
	{		
		do 
		{
			dim = dim + 1 ;
			strtod( buf, &temp ) ;
			strcpy( buf, temp ) ;
			chg = var - strlen(buf) ;
			var = var - chg ;
		}
		while ( 0 != chg ) ;
	}
	else
	{ 
		fclose( smo_stream );
		printf("the first line in the file is empty.\n") ;
		return 1 ;
	}

	if ( 0 > dim ) 
	{
		fclose( smo_stream );

#ifdef SMO_DEBUG
		printf( "input dimension is less than one.\n") ;
#endif
		return 1 ;
	}

	if (inputdim > 0)
	{
		if (inputdim == dim + 1 ) // test file without target
		{
			// try to open "*target*.*" as target
			// create target file name
			pstr = strstr( inputfilename , "test" ) ;
			if (NULL != pstr)
			{
				result = abs( inputfilename - pstr ) ;
				strncpy (buf, inputfilename, result ) ;
				buf[result] = '\0' ;
				strcat(buf, "targets") ;
				strcat (buf, pstr+4) ;
				smo_target = fopen( buf, "r+t" ) ;
			}
			dim = inputdim ;
			pairs->dimen = dim ;
		}
		else if ( inputdim != dim )
		{
			printf("Dimensionality in testdata is inconsistent with traindata.\n") ;
			return 1 ;
		}
		else
			pairs->dimen = dim ;
	}
	else
		pairs->dimen = dim ;
	
	//initialize the x_mean and x_devi in Data_List pairs

	if ( NULL == (pairs->x_mean = (double *)(malloc(dim*sizeof(double))) ) \
		|| NULL == (pairs->x_devi = (double *)(malloc(dim*sizeof(double))) ) \
		|| NULL == (xmean = (double *)(malloc(dim*sizeof(double))) ) )
	{		
		if (NULL != pairs->x_mean) 
			free(pairs->x_mean) ;
		if (NULL != pairs->x_devi) 
			free(pairs->x_devi) ;
		if (NULL != xmean)
			free(xmean) ;
		if (NULL != smo_target)
			fclose( smo_target ) ;
		if (NULL != smo_stream)
			fclose( smo_stream );
		return 1 ;
	}
	for ( j = 0; j < dim; j ++ )
		pairs->x_mean[j] = 0 ;
	for ( j = 0; j < dim; j ++ )
		pairs->x_devi[j] = 0 ;
	for ( j = 0; j < dim; j ++ )
		xmean[j] = 0 ;

	// begin to initialize data_list for digital input only
	printf("\nLoading samples from %s ...  \n", inputfilename) ;
	pairs->datatype = PREFERENCE ; 

	rewind( smo_stream ) ;
	fgets( buf, LENGTH, smo_stream ) ;
	do
	{

#ifdef SMO_DEBUG 
		printf("%d\n", index) ;
		printf("%s\n\n\n", buf) ;
#endif
		point = (double *) malloc( (dim+1) * sizeof(double) ) ; // Pairs to free them
		if ( NULL == point )
		{
			printf("not enough memory.\n") ;
			if (NULL != smo_target)
				fclose( smo_target ) ;
			if (NULL != smo_stream)
				fclose( smo_stream );
			if (NULL != pairs->x_mean) 
				free(pairs->x_mean) ;
			if (NULL != pairs->x_devi) 
				free(pairs->x_devi) ;
			if (NULL != xmean)
				free(xmean) ;
			Clear_Data_List( pairs ) ;
			return 1 ;
		}
		var = strlen( buf ) ;	
		i = 0 ;
		chg = dim ;

		while ( chg>0 && i<dim)
		{
			point[i] = strtod( buf, &temp ) ;
			i++ ;
			strcpy( buf, temp ) ;
			chg = var - strlen(buf) ;
			var = var - chg ;
		}
		point[dim]=0 ;
		if (i==dim && chg>0 && var>=0)
			//y = (int)strtod( buf, &temp ) ;
			y = 0 ;
		else
		{
			free(point) ;
			//y = 0 ;
			printf("Warning: the input file contains a blank or defective line %lu.\n",index ) ;
			//exit(1);
		}
		// load y as target from other file when dim+1
		if (NULL != smo_target)
		{
			if ( NULL != fgets( buf, LENGTH, smo_target ) )
			{
				var = strlen( buf ) ;
				y = (int)strtod( buf, &temp ) ;
				strcpy( buf, temp ) ;
				chg = var - strlen(buf) ;
				if (chg==0)
					printf("Warning: the target file contains a blank line.\n") ;
			}
			else
				printf("Warning: the target file is shorter than the input file.\n") ;
		}

		/*	for ( i = 0; i < dim; i ++ )
			{
				point[i] = strtod( buf, &temp ) ;
				strcpy( buf, temp ) ;
			}
			y = strtod( buf, &temp ) ;

			// load y as target from other file when dim+1
			if (NULL != smo_target)
			{
				fgets( buf, LENGTH, smo_target ) ;
				y = strtod( buf, &temp ) ;
			}*/

		if (chg>0) 
		{							
			if ( 0 == Add_Data_List( pairs, Create_Data_Node(index, point, y) ) )
			{
				// update statistics
				pairs->mean = (mean * (((double)(pairs->count)) - 1) + y )/ ((double)(pairs->count))  ;
				pairs->deviation = pairs->deviation + (y-mean)*(y-mean) * ((double)(pairs->count)-1)/((double)(pairs->count));			
				mean = pairs->mean ;	
				for ( j=0; j<dim; j++ )
				{
					pairs->x_mean[j] = (xmean[j] * (((double)(pairs->count)) - 1) + point[j] )/ ((double)(pairs->count))  ;
					pairs->x_devi[j] = pairs->x_devi[j] + (point[j]-xmean[j])*(point[j]-xmean[j]) * ((double)(pairs->count)-1)/((double)(pairs->count));			
					xmean[j] = pairs->x_mean[j] ;
				}
				if (y>ymax)
				{ ymax = y ; pairs->i_ymax = index ;}
				if (y<ymin)
				{ ymin = y ; pairs->i_ymin = index ;}
				
				// check data type 
				Add_Label_Data_List( &label, Create_Data_Node(index, point, y) ) ;
				index ++ ;
			}
			else
			{
#ifdef SMO_DEBUG 
				printf("%d\n", index) ;
				printf("duplicate data \n") ;
#endif
			}
		}
	}
	while( !feof( smo_stream ) && NULL != fgets( buf, LENGTH, smo_stream ) ) ;

	if (label.count>=2)
		pairs->datatype = ORDINALREGRESSION ;
	//else
	//	printf("Warning : not a ordinal regression.\n") ;

	if (pairs->count < MINNUM || (pairs->datatype == UNKNOWN && inputdim == 0 ) ) 
	{
		printf("too few input pairs\n") ;
		Clear_Data_List( pairs ) ;
		if (NULL != pairs->x_mean) 
			free(pairs->x_mean) ;
		if (NULL != pairs->x_devi) 
			free(pairs->x_devi) ;
		if (NULL != xmean)
			free(xmean) ;
		if (NULL != smo_target)
			fclose( smo_target ) ;
		if (NULL != smo_stream)
			fclose( smo_stream );
		return 1 ;
	}
	
	pairs->featuretype = (int *) malloc(pairs->dimen*sizeof(int)) ;                                             
	if (NULL != pairs->featuretype)                                                                             
	{                                                                                                           
		//default 0                                                                                         
		for (sz=0;sz<pairs->dimen;sz++)                                                                     
			pairs->featuretype[sz] = 0 ;                                                                
		                                                                                                    
		if (0==inputdim)                                                                                    
			pstr = strstr( inputfilename, "train") ;	// 46                                       
		else                                                                                                
			pstr = strstr( inputfilename, "test") ;	// 46                                               
		if (NULL != pstr)                                                                                   
		{                                                                                                   
			sz = abs( pstr - inputfilename ) ;                                                          
			pstr = strrchr( inputfilename, '.') ;	// 46                                               
			strncpy( buf, inputfilename, sz ) ;                                                         
			buf[sz]='\0' ;                                                                              
			strcat( buf, "feature" ) ;			                                            
			strcat( buf, pstr ) ;                                                                       
			fid = fopen(buf,"r+t") ;                                                                    
			if (NULL != fid)                                                                            
			{                                                                                           
				printf("Loading the specifications of feature type in %s ...",buf) ;                
				sz = 0 ;                                                                            
				while (!feof(fid) && NULL!=fgets(buf,LENGTH,fid) )                                  
				{                                                                                   
					i=strlen(buf) ;                                                             
					if (i>1)                                                                    
					{                                                                           
						if (sz>=pairs->dimen)                                               
						{                                                                   
							printf("Warning : feature type file is too long.\n") ;      
							sz = pairs->dimen-1 ;                                       
						}                                                                   
						pairs->featuretype[sz] = atoi(buf) ;                                
						sz += 1 ;                                                           
					}                                                                           
					else                                                                        
						printf("Warning : blank line in feature type file.\n") ;            
				}                                                                                   
				if (sz!=pairs->dimen)                                                               
				{                                                                                   
					//default 0                                                                 
					for (sz=0;sz<pairs->dimen;sz++)                                             
						pairs->featuretype[sz] = 0 ;                                        
					printf(" RESET as default.\n") ;                                            
				}                                                                                   
				else                                                                                
					printf(" done.\n") ;                                                        
				fclose(fid) ;                                                                       
			}                                                                                           
		}                                                                                                   
	}                                                                                                           

	pairs->deviation = sqrt( pairs->deviation / ((double)(pairs->count - 1.0)) ) ;
	for ( j=0; j<dim; j++ )
		pairs->x_devi[j] = sqrt( pairs->x_devi[j] / ((double)(pairs->count - 1.0)) ) ;	
	
	// set target value as +1 or -1, if data type is CLASSIFICATION
	if ( UNKNOWN != pairs->datatype && 0 == inputdim )
	{
			node = pairs->front ;
			while ( node != NULL )
			{				
				node = node->next ; 
			}
			pairs->deviation = 1.0 ;
			pairs->mean = 0 ;
			pairs->normalized_output = 0 ;
	}

	for ( j=0; j<dim; j++ )
	{
		if (pairs->featuretype[j] != 0)
		{
			pairs->x_devi[j] = 1 ;
			pairs->x_mean[j] = 0 ;
		}
	}
	
    if (inputdim>0) // do not normailize data for TESTING
	{
		pairs->normalized_output = 0 ;
		pairs->normalized_input = 0 ; 
	}

	// normalize the target if needed 
	node = pairs->front ;
	while ( node != NULL )
	{
		if ( 1 == pairs->normalized_input )
		{
			for ( j=0; j<dim; j++ )
			{				
				if (pairs->x_devi[j]>0)
					node->point[j] = (node->point[j]-pairs->x_mean[j])/(pairs->x_devi[j]) ;
				else
					node->point[j] = 0 ;
			}
		}
		node = node->next ; 
	}
	printf("Total %lu samples with %lu dimensions for ", pairs->count, pairs->dimen) ;	

	if	(inputdim > 0)
		printf("TESTING.\r\n") ;
	else 
	{
		if( PREFERENCE == pairs->datatype )
		{
			printf("PREFERENCE.\r\n") ;
			pairs->classes = 1 ;
		}
		else if ( ORDINALREGRESSION == pairs->datatype )
		{
			printf("ORDINAL %lu REGRESSION.\r\n",label.count) ;
			pairs->classes = label.count ;
			if (NULL != pairs->labels)
				free( pairs->labels ) ;
			i=0;
			pairs->labels = (int*)malloc(pairs->classes*sizeof(int)) ;
			pairs->labelnum = (int*)malloc(pairs->classes*sizeof(int)) ;
			if (NULL != pairs->labels&&NULL != pairs->labelnum)
			{
				node = label.front ;
				j=0 ;				
				printf("ordinal varibles : ") ;
				while (NULL!=node)
				{
                                       	if (node->target<1 || node->target>pairs->classes || node->target!=(unsigned int)j+1)
                                       	{
                                               	printf("\nError : samples should be sorted with target from 1 to %d.\n",(int)pairs->classes) ;
                                               	exit(1) ;
                                       	}
                                       	pairs->labels[node->target-1] = node->target ;
                                       	if (node->target-1==0)
                                               	t0 = node->target ;
                                       	if (node->target==(int)pairs->classes)
                                               	tr = node->target ;
                                       	pairs->labelnum[node->target-1] = node->fold ;
                                       	i += node->fold ;
                                       	printf("%d(%d)  ", node->target, node->fold) ;
                                       	node = node->next ;
					j+=1 ;
				}
				printf("\n") ;
				if (i!=(int)pairs->count||t0!=1||tr!=(int)pairs->classes)
				{
					printf("Error in data list.\n") ;
					exit(1) ;
				}
			}
			else
			{
				printf("fail to malloc for pairs->labels.\n") ;			
				exit(1) ;
			}
		}
		else 
			printf("UNKNOWN.\r\n") ;
	}
	if (1 == pairs->normalized_input)
		printf("Inputs are normalized.\r\n") ;
	
	if (1 == pairs->normalized_output && pairs->deviation > 0)
		printf("Outputs are normalized.\r\n") ;
	if ( inputdim > 0 && pairs->deviation <= 0 )
		printf("Tragets are not at hand.\r\n") ;

#ifdef _SOFTMAX_SERVER	
	pairs->classes = 3 ;
	pairs->labels[0] = 2 ;
	pairs->labelnum[0] = 0 ;
	pairs->labels[1] = 0 ;
	pairs->labelnum[1] = 0 ;
	pairs->labels[2] = 1 ;
	pairs->labelnum[2] = 0 ;
#endif

	Clear_Label_Data_List ( &label ) ;
	if (NULL != smo_target)
		fclose( smo_target ) ;
	if (NULL != smo_stream)
		fclose( smo_stream );
	if ( NULL != xmean )
		free( xmean ) ;
	return 0 ;
}

// end of loadfile.c
