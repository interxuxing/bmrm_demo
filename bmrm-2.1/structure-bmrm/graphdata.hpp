#ifndef _GRAPHDATA_HPP_
#define _GRAPHDATA_HPP_

#include "common.hpp"
#include "sml.hpp"
#include "data.hpp"
#include <vector>
#include <iostream>


#define FRACTION 8
#define MAXLOSS 18
#define Num_Label 260
#define MAXLABELS 20
#define FEATURESIZE 37152
#define Num_Train 4500
#define Num_Test 499
#define NCOPIES 1
extern int FEATURESIZE;
#define NFEATURES (NCOPIES*FEATURESIZE)
#define NNEIGHBOURS 5

#define Scalar double

#include "mkl.h"
#include "mkl_cblas.h"
#include <boost/numeric/ublas/matrix.hpp>

/**
 A fast sorting algorithm.
 Courtesy of http://warp.povusers.org/SortComparison/
 */

#define COMP >

template<typename ItemType>
void InsertionSort(ItemType* array, unsigned size, unsigned int* pa)
{
  for(unsigned i = 1; i < size; ++i)
  {
    ItemType val = array[i];
    int vali = pa[i];
    unsigned j = i;
    while(j > 0 && val COMP array[j-1])
    {
      array[j] = array[j-1];
      pa[j] = pa[j-1];
      --j;
    }
    array[j] = val;
    pa[j] = vali;
  }
}

template<typename ItemType>
unsigned Partition(ItemType* array, unsigned f, unsigned l, const ItemType& pivot, unsigned int* pa)
{
  unsigned i = f-1, j = l+1;
  while(true)
  {
    while(pivot COMP array[--j]);
    while(array[++i] COMP pivot);
    if(i<j)
    {
      ItemType tmp = array[i];
      array[i] = array[j];
      array[j] = tmp;
      int tmpi = pa[i]; pa[i] = pa[j]; pa[j] = tmpi;
    }
    else
      return j;
  }
}

template<typename ItemType>
void QuickSortImpl(ItemType* array, unsigned f, unsigned l, unsigned int* pa)
{
  while(f < l)
  {
    unsigned m = Partition(array, f, l, array[f], pa);
    QuickSortImpl(array, f, m, pa);
    f = m+1;
  }
}

template<typename ItemType>
void QuickSort(ItemType* array, unsigned size, int* pa)
{
  QuickSortImpl(array, 0, size-1, pa);
}

template<typename ItemType>
void MedianHybridQuickSortImpl(ItemType* array, unsigned f, unsigned l, unsigned int* pa)
{
  while(f+16 < l)
  {
    ItemType v1 = array[f], v2 = array[l], v3 = array[(f+l)/2];
    ItemType median =
      v1 < v2 ?
      ( v3 < v1 ? v1 : std::min(v2, v3)
        ) :
      ( v3 < v2 ? v2 : std::min(v1, v3)
        );
    unsigned m = Partition(array, f, l, median, pa);
    MedianHybridQuickSortImpl(array, f, m, pa);
    f = m+1;
  }
}

template<typename ItemType>
void MedianHybridQuickSort(ItemType* array, unsigned size, unsigned int* pa)
{
  MedianHybridQuickSortImpl(array, 0, size-1, pa);
  InsertionSort(array, size, pa);
}

class CGraphData: public CData
{
  public:
    // constructor
    CGraphData();
    virtual ~CGraphData();
    int _N;

    // The features and labels for every image.
    float** imageMatrix;
    float* devPtrF;
    std::vector<int> imageLabels;
    int** imageMultiLabels; // 大小为nImage X MAXLABELS(20)
    int** latentLabels; // 大小为nImage X NNEIGHBOURS(5)

    int** lossMatrix; // The loss matrix is always integer valued in this dataset.
    float** indicator;

    unsigned int numOfExample;
    unsigned int numOfAllExample;

    virtual bool bias(void) const { return false; }
    virtual unsigned int dim(void) const { return NFEATURES; }

    virtual unsigned int slice_size(void) const { return _N; }
    virtual unsigned int size(void) const { return _N; }

  private:
    CGraphData(const CGraphData&);
};

#endif
