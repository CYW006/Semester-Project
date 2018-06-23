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

int Create_Pref_List(Pref_List * list)
{
	if (NULL == list)
		return 1 ;
	list->count=0;
	list->front=NULL;
	list->rear=NULL;
	return 0 ;
}
Pref_Node * Create_Pref_Node(unsigned long int index)
{
	Pref_Node * node ;
	if (index<1)
		return NULL ;
	node = (Pref_Node *) malloc(sizeof(Pref_Node)) ;
	if (NULL == node)
		return node ;
	node -> index = index ;
	node -> next = NULL ;
	return node ;
}
int Is_Pref_Empty(Pref_List * list)
{
	if (NULL == list) 
		return 1 ;
	if ((list -> front == NULL) && (list -> rear == NULL))
		return  0 ;
	else 
		return  1;
}
int Add_Pref_List(Pref_List * list, Pref_Node * node)
{
	//Pref_Node * temp = NULL ;
	if ( NULL == node || NULL == list)
		return 1 ;

	if (!Is_Pref_Empty(list))
		list -> front = list -> rear = node ;	
	else
	{
		list -> rear -> next = node ;
		list -> rear = node ;
	}
	list -> count += 1 ;
    return 0 ;
}
int Clear_Pref_List(Pref_List * list)
{
	Pref_Node * node ;
	Pref_Node * temp ;
	if (NULL ==list)
		return 0 ;
	node = list->front ;
	while (NULL!=node)
	{
		temp=node ;
		node=node->next ;
		//printf("%ld: u=%ld v=%ld\n",temp->index,temp->u,temp->v) ;
		free(temp) ;		
		list->count -= 1 ;
	}
	list->front = NULL ;
	list->rear = NULL ;

	if (list->count!=0)
	{
		printf("Error in the Pref_List.\n");
		list->count = 0 ;
	}
	return 0 ;
}

int Pref_Loadpair ( Pref_List * pref, char * inputfilename ) 
{
	FILE * fid ;
	unsigned long int index = 0 ;
	Pref_Node * node ;
	char buf[LENGTH] ;
	char * temp ;

	if (NULL==pref || NULL==inputfilename)
		return 1 ;

	fid = fopen(inputfilename,"r+t") ;
	if (NULL == fid)
		return 1 ;

	printf("\nLoading preference pairs from %s ...  \n", inputfilename) ;

	rewind( fid ) ;
	fgets( buf, LENGTH, fid ) ;
	if (strlen( buf ) < 1)
	{
		fclose(fid) ;
		return 1 ;
	}
	do
	{
		index += 1 ;
		node = Create_Pref_Node (index) ;
		node->u = strtol( buf, &temp, 10 ) ;
		strcpy( buf, temp ) ;
		node->v = strtol( buf, &temp, 10 ) ;
		Add_Pref_List( pref, node) ;
	}
	while( !feof( fid ) && NULL != fgets( buf, LENGTH, fid ) ) ;

	printf("%ld pairs loaded for training.\n", pref->count) ;
	fclose(fid) ;

	return 0 ;
}

// end of loadpair.c
