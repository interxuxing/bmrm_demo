#ifndef _GRAPHMATCHLOSS_CPP_
#define _GRAPHMATCHLOSS_CPP_

#include "graphmatchloss.hpp"
#include "configuration.hpp"
#include <fstream>
#include <sstream>

#include "math.h"

#include "ctime"
#include "limits.h"

#include <map>

#include <omp.h>

#include "mkl.h"
#include "mkl_boost_ublas_matrix_prod.hpp"
#include <boost/numeric/ublas/matrix.hpp>

#include "cublas.h"

#include "radixsort.hpp"
#include "cudpp/cudpp.h"
using namespace boost::numeric::ublas;
using namespace std;

/**
 * How similar are the features of this image to the representative features of the class?
 * imagePhi ����feature��indicator(\theta_binary)��hadamard product�����ص�res���������
 */
void imagePhi(float* imFeatures, float* indicator, float* res)
{
  for (int c = 0; c < NCOPIES; c ++)
    for (int i = 0; i < FEATURESIZE; i ++)
      *(res++) = imFeatures[i] * *(indicator++);
}

bool sortFunc(pair<Scalar,int> a, pair<Scalar,int> b)
{
  return a.first > b.first;
}
/*
 * void assignment��������������㹫ʽ12�� ��ȡ\bar{Y} (��������\hat{Y})��Ӧlabel�������� ���浽classesbar��
*/
void assignment(float* imFeatures, Scalar* w, float** indicator, int* classes, int maxviolator, int label, int** lossMatrix, float* preProduct, unsigned int* prePa)
{
  float* phi = new float [NFEATURES];
  
  unsigned int* pa = prePa;
  
  if (!pa)
  {
    pa = new unsigned int [NCLASSES];
    for (int i = 0; i < NCLASSES; i ++)
      pa[i] = i;
  }
  
  float* innerProducts = preProduct;
  if (!innerProducts)
  {
    innerProducts = new float [NCLASSES];
    for (int i = 0; i < NCLASSES; i ++)
    {
      imagePhi(imFeatures, indicator[i], phi); //imagePhi��������hardmard�ڻ��� ���ص�phi�У�phi��һ��NFEATUREά�ȵ�����
      float innerProduct = 0;
      if (w)
        for (int j = 0; j < NFEATURES; j ++) innerProduct += phi[j]*w[j];//�����Ǽ���phi*w, �µ�joint feature vectorֵ
      else
        for (int j = 0; j < NFEATURES; j ++) innerProduct += phi[j];
      innerProducts[i] = innerProduct;
    }
  }

  if (!prePa)
    MedianHybridQuickSort<float>(innerProducts, NCLASSES, pa); //��1000��classes���ڻ�ֵ���� ��������

  if (!maxviolator)
  {
    for (int i = 0; i < NNEIGHBOURS; i ++)
      classes[i] = pa[i]; //ֻȡǰ5������Ϊclass����� (���û���趨maxviolator)
  }
  else // ����趨��maxviolator
  {
    Scalar maxVal = -numeric_limits<Scalar>::max();
    Scalar sumPhi = numeric_limits<Scalar>::max();
    int* witnesses = new int [NNEIGHBOURS];
    int firstInd = 0;
    for (int L = 0; L <= MAXLOSS; L ++)
    {
      if (L + sumPhi < maxVal) continue; // Since sumPhi decreases monotonically, this value of L cannot result in the best solution.
      int ind = firstInd; // The inner-products are in decreasing order, so the ones used for L will be no bigger than the ones used for L-1.
      int n = 0;
      while (n < NNEIGHBOURS and ind < NCLASSES)
      { // Choose the labels with the largest inner-products, subject to the constraint that the loss is greater than L.
    	  // ѡ��ӵ��inner-product����labels, ������constraint����L
        if (lossMatrix[label][pa[ind]] >= L) //����label��classes�е�pa[ind] (1000�е�ǰK��)��\deltaֻ���Ƿ����L�� ������ڣ�����ӵ�witnesses�У�
        {
          if (n == 0) firstInd = ind;
          witnesses[n++] = ind;
        }
        ind ++;
      }
      if (n < NNEIGHBOURS) break;
      int minLoss = MAXLOSS;
      sumPhi = 0;
      for (int n = 0; n < NNEIGHBOURS; n ++)
      { // Find the minimum loss amongst the chosen labels. If the minimum loss is M, then we needn't consider solutions for L < M.
        sumPhi += innerProducts[witnesses[n]];
        int clazz = pa[witnesses[n]]; //pa��1000��inner-product���������
        if (lossMatrix[label][clazz] < minLoss) minLoss = lossMatrix[label][clazz];
      }
      if (minLoss < L)
      {
        printf("Got impossible loss\n");
        exit(0);
      }
      L = minLoss;
      if (L + sumPhi > maxVal)
      {
        maxVal = L + sumPhi;
        for (int n = 0; n < NNEIGHBOURS; n ++)
          classes[n] = pa[witnesses[n]]; //���ֵ��classes�� ��Ϊ\bar{Y}^n�е�����������
      }
    }
    delete [] witnesses;
  }

  if (!prePa) delete [] pa;
  if (!preProduct) delete [] innerProducts;
  delete [] phi;
}

/**
 * Compute the feature std::vector.
 */
void CGraphMatchLoss::Phi(float* imFeatures, int* classes, int N, int* latent, int L, float* res)
{
  float* phi = new float [NFEATURES];
  for (int i = 0; i < NFEATURES; i ++) res[i] = 0;

  for (int c = 0; c < N; c ++)
  {
    imagePhi(imFeatures, _data->indicator[classes[c]], phi); //N=1�� �����Ǽ��㹫ʽ10�е�\Phi(x^n, Y^n)
    for (int i = 0; i < NFEATURES; i ++)
      res[i] += phi[i];
  }
  
  for (int l = 0; l < L; l ++) //L = K-1
  {
    imagePhi(imFeatures, _data->indicator[latent[l]], phi); //����latent[l]������Z_n�е�����, ������㹫ʽ10�е�\Phi(x^n, Z^n)
    for (int i = 0; i < NFEATURES; i ++)
      res[i] += phi[i];
  }
  
  delete [] phi;
}

/** Constructor. */
CGraphMatchLoss::CGraphMatchLoss(CModel* &model, CGraphData* &data) : CLoss(model, 1, NFEATURES, data->bias()), _data(data)
{
  Configuration &config = Configuration::GetInstance();
  verbosity = config.GetInt("Loss.verbosity");
}

/** Destructor. */
CGraphMatchLoss::~CGraphMatchLoss() {}

void CGraphMatchLoss::Usage() {}

int iteration = 0;

unsigned int** newMatrixU(int x, int y)
{
  unsigned int* matflat = new unsigned int [x*y];
  unsigned int** matsquare = new unsigned int* [x];
  for (int i = 0; i < x; i ++)
    matsquare[i] = matflat + i*y;

  return matsquare;
}

void CGraphMatchLoss::ComputeLossAndGradient(Scalar& loss, TheMatrix& grad)
{
  TheMatrix &w = _model->GetW();
  grad.Zero();

  Scalar* dat = w.Data();
  Scalar* raw_g = grad.Data();

  loss = 0;
  for (int i = 0; i < NFEATURES; i ++) raw_g[i] = 0;

  int nthreads = 4;
  Scalar* lossi = new Scalar [nthreads];
  Scalar** raw_gi = new Scalar* [nthreads];

  for (int t = 0; t < nthreads; t ++)
  {
    lossi[t] = 0;
    raw_gi[t] = new Scalar [NFEATURES];
    for (int j = 0; j < NFEATURES; j ++)
      raw_gi[t][j] = 0;
  }

  float* indicatorW = new float [NCLASSES*NFEATURES];
  for (int i = 0; i < NFEATURES; i ++)
    for (int j = 0; j < NCLASSES; j ++)
      indicatorW[i*NCLASSES + j] = _data->indicator[j][i] * dat[i];

  float* preProduct = new float [_data->_N * NCLASSES];
  float** preProductFlat = new float* [_data->_N];
  for (int i = 0; i < _data->_N; i ++)
    preProductFlat[i] = preProduct + i*NCLASSES;

  float* devPtrW;
  float* devPtrI;
  
  if (cublasAlloc(NCLASSES*_data->_N, sizeof(*preProduct), (void**)&devPtrI) != CUBLAS_STATUS_SUCCESS) { printf("Failed to allocate matrix I\n"); exit(1); }
  if (cublasSetMatrix(NCLASSES, _data->_N, sizeof(*preProduct), preProduct, NCLASSES, devPtrI, NCLASSES) != CUBLAS_STATUS_SUCCESS) { printf("Failed to copy matrix I\n"); exit(1); }
  if (cublasAlloc(NCLASSES*NFEATURES, sizeof(*indicatorW), (void**)&devPtrW) != CUBLAS_STATUS_SUCCESS) { printf("Failed to allocate matrix W\n"); exit(1); }
  if (cublasSetMatrix(NCLASSES, NFEATURES, sizeof(*indicatorW), indicatorW, NCLASSES, devPtrW, NCLASSES) != CUBLAS_STATUS_SUCCESS) { printf("Failed to copy matrix W\n"); exit(1); }

  cublasSgemm('N', 'N', NCLASSES, _data->_N, NFEATURES, 1.0, devPtrW, NCLASSES, _data->devPtrF, NFEATURES, 0, devPtrI, NCLASSES);

  /*
  Change the above alpha to -1...

  // Sorting on the GPU (turns out to be slow).
  unsigned int** h_values = newMatrixU(_data->_N, NCLASSES);
  for (int i = 0; i < _data->_N; i ++)
    for (int j = 0; j < NCLASSES; j ++)
      h_values[i][j] = j;
    
  unsigned int* d_values;
  cudaMalloc((void **)&d_values, _data->_N*NCLASSES*sizeof(unsigned int));
  cudaMemcpy(d_values, *h_values, _data->_N*NCLASSES*sizeof(unsigned int), cudaMemcpyHostToDevice);

  for (int i = 0; i < _data->_N; i ++)
  {
    float* d_keysi = devPtrI + i*NCLASSES;
    unsigned int* d_valuesi = d_values + i*NCLASSES;
    nvRadixSort::RadixSort radixsort(NCLASSES, false);
    radixsort.sort(d_keysi, d_valuesi, NCLASSES, 32, true);
  }
  
  cudaMemcpy(*h_values, d_values, _data->_N*NCLASSES*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cublasFree(d_values);
  
  cublasSscal(_data->_N*NCLASSES, -1.0, devPtrI, 1);
  */

  cublasGetMatrix(NCLASSES, _data->_N, sizeof(*preProduct), devPtrI, NCLASSES, preProduct, NCLASSES);
  cublasFree(devPtrW);
  cublasFree(devPtrI);

  #pragma omp parallel for
  for (int i = 0; i < _data->_N; i ++)
  {
    int threadnum = omp_get_thread_num();
    int yLabel = _data->imageLabels[i];

    int* classesbar = new int [NNEIGHBOURS];
    float* resy = new float [NFEATURES];
    float* resybar = new float [NFEATURES];
    // assignment��������������㹫ʽ12�� ��ȡ\bar{Y} (��������\hat{Y})��Ӧlabel�������� ���浽classesbar��
    assignment(_data->imageMatrix[i], dat, _data->indicator, classesbar, 1, yLabel, _data->lossMatrix, preProductFlat[i], NULL);
    // ���Phi()�����������еĹ�ʽ10���õ���\Phi(x^n, \Omega^n)
    Phi(_data->imageMatrix[i], &yLabel, 1, _data->latentLabels[i], NNEIGHBOURS - 1, resy);
    // ���������Ԥ���\bar{Y},�õ���\Phi(x^n, \bar{Y})
    Phi(_data->imageMatrix[i], classesbar, NNEIGHBOURS, NULL, 0, resybar);
    // ���������label loss, ������\bar{Y}, Y^n, ���㹫ʽ3
    Scalar labloss = LabelLoss(yLabel, classesbar, NNEIGHBOURS);

    lossi[threadnum] += labloss;
    for (int j = 0; j < NFEATURES; j ++)
    {
      lossi[threadnum] += dat[j]*(resybar[j]-resy[j]); //���չ�ʽ14�� ���������loss���֣� <\Phi(x^n, \bar{Y}^n),\theta> - <\Phi(x^n - \Omega^n), \theta>
      raw_gi[threadnum][j] += (1.0/_data->_N)*(resybar[j]-resy[j]); //���չ�ʽ15�� ����������ݶȲ���<\Phi(x^n, \bar{Y}^n),\theta> - <\Phi(x^n - \Omega^n), \theta>
    }

    delete [] classesbar;
    delete [] resy;
    delete [] resybar;
  }
  
  delete [] indicatorW;
  delete [] preProduct;
  delete [] preProductFlat;
  
  /*
  delete [] *h_values;
  delete [] h_values;
  */
  
  for (int t = 0; t < nthreads; t ++)
  {
    loss += lossi[t];
    for (int k = 0; k < NFEATURES; k ++)
      raw_g[k] += raw_gi[t][k];

    delete [] raw_gi[t];
  }
  delete [] lossi;
  delete [] raw_gi;

  loss = loss/_data->_N;
}

void CGraphMatchLoss::Predict(CModel *model)
{
  Evaluate(model);
}

/**
 * Compare the performance of the model after learning to the model before learning.
 * The performance before learning is found using a constant weight std::vector.
 */
void CGraphMatchLoss::Evaluate(CModel *model)
{
  TheMatrix &w = _model->GetW();
  Scalar* loss_noweight = new Scalar [NNEIGHBOURS];
  Scalar* loss_weight = new Scalar [NNEIGHBOURS];
  
  for (int i = 0; i < NNEIGHBOURS; i ++)
  {
    loss_noweight[i] = 0;
    loss_weight[i] = 0;
  }

  Scalar* dat = w.Data();

  int nthreads = 4;
  Scalar** loss_noweights = new Scalar* [nthreads];
  Scalar** loss_weights = new Scalar* [nthreads];
  for (int i = 0; i < nthreads; i ++)
  {
    loss_noweights[i] = new Scalar [NNEIGHBOURS];
    loss_weights[i] = new Scalar [NNEIGHBOURS];
    for (int j = 0; j < NNEIGHBOURS; j ++)
    {
      loss_noweights[i][j] = 0;
      loss_weights[i][j] = 0;
    }
  }
  
//   FILE* labels = fopen("labels.txt", "w");
//   FILE* prediction = fopen("prediction.txt", "w");

  #pragma omp parallel for
  for (int i = 0; i < _data->_N; i ++)
  {
    int yLabel = _data->imageLabels[i];

    int* classes0 = new int [NNEIGHBOURS];
    int* classes1 = new int [NNEIGHBOURS];

    assignment(_data->imageMatrix[i], NULL, _data->indicator, classes0, 0, yLabel, _data->lossMatrix, NULL, NULL);
    assignment(_data->imageMatrix[i], dat, _data->indicator, classes1, 0, yLabel, _data->lossMatrix, NULL, NULL);
    
//     for (int n = 0; n < NNEIGHBOURS; n ++)
//       fprintf(prediction, "%d ", classes0[n] + 1);
//     fprintf(prediction, "\n");
//     fprintf(labels, "%d\n", yLabel + 1);

    for (int n = 1; n <= NNEIGHBOURS; n ++)
    {
      loss_noweights[omp_get_thread_num()][n-1] += LabelLoss(yLabel, classes0, n);
      loss_weights[omp_get_thread_num()][n-1] += LabelLoss(yLabel, classes1, n);
    }

    delete [] classes0;
    delete [] classes1;
  }

  for (int i = 0; i < nthreads; i ++)
  {
    for (int j = 0; j < NNEIGHBOURS; j ++)
    {
      loss_noweight[j] += loss_noweights[i][j];
      loss_weight[j] += loss_weights[i][j];
    }
  }
  
  for (int i = 0; i < NNEIGHBOURS; i ++)
  {
    loss_noweight[i] /= _data->_N;
    loss_weight[i] /= _data->_N;
  }
  
  for (int i = 0; i < NNEIGHBOURS; i ++)
  {
    cout << "Error @" << i + 1 << endl;
    cout << "  without weights: " << loss_noweight[i] << endl;
    cout << "   with weights: " << loss_weight[i] << endl;
  }
  
  for (int i = 0; i < nthreads; i ++)
  {
    delete [] loss_noweights[i];
    delete [] loss_weights[i];
  }
  delete [] loss_noweights;
  delete [] loss_weights;
  delete [] loss_noweight;
  delete [] loss_weight;
}

/** Loss function. */
Scalar CGraphMatchLoss::LabelLoss(int label, int* classes, int N)
{
  Scalar l = 0;
  l = _data->lossMatrix[label][classes[0]];
  for (int i = 1; i < N; i ++)
    if (_data->lossMatrix[label][classes[i]] < l) l = _data->lossMatrix[label][classes[i]];

  return l;
  // ����label��ground-truth��һ��single label�� classes����5������ֵ����ʾԤ���Y�е�5��label, loss��ѡ����label��˵��
  //	��С���Ǹ�y\in{Y}
}

#endif


void assignment(float* imFeatures, Scalar* w, float** indicator, int* classes, int maxviolator, int* gtlabels, int** lossMatrix, float* preProduct, unsigned int* prePa)
{
	float* phi = new float [NFEATURES];
  
	unsigned int* pa = prePa;
  
	if (!pa) //
	{
		pa = new unsigned int [NCLASSES];
		for (int i = 0; i < NCLASSES; i ++)
		  pa[i] = i;
	}
  
	float* innerProducts = preProduct;
	if (!innerProducts) // �@��?���I�V�O�I��?preProduct�s����NULL
	{
		innerProducts = new float [NCLASSES];
		for (int i = 0; i < NCLASSES; i ++)
		{
		  imagePhi(imFeatures, indicator[i], phi); //imagePhi�p��?�Zhardmard��?�C �ԉ�phi���Cphi���꘢NFEATURE?�x�I����
		  float innerProduct = 0;
		  if (w)
			for (int j = 0; j < NFEATURES; j ++) innerProduct += phi[j]*w[j];//?����?�Zphi*w, �V�Ijoint feature vector?
		  else
			for (int j = 0; j < NFEATURES; j ++) innerProduct += phi[j];
		  innerProducts[i] = innerProduct;
		}
	}
  
	if (!prePa) // �@��prePa == NULL
		MedianHybridQuickSort<float>(innerProducts, NCLASSES, pa); //?1000��classes�I��??�r���C �~���r��
	
	if (!maxviolator) // maxviolator = 0
	{
		for (int i = 0; i < NNEIGHBOURS; i ++)
		  classes[i] = pa[i]; //����O5���C��?class�I?�o (�@�ʖv�L?��maxviolator)
	}
	else
	{
		Scalar maxVal = -numeric_limits<Scalar>::max();
		Scalar sumPhi = numeric_limits<Scalar>::max();
		int* witnesses = new int [NNEIGHBOURS]; // witnesses�p����?��Iy_\bar
		int firstInd = 0, gtInd = -1;
		int Lc = 0, Ly = 0;

		for(int c = 0; c < NCLASSES; c++)
		{
			// compare the loss of sum(c, gtlabels)
			for(int yn = 0; yn < MAXLABELS; yn++)
			{
				gtInd = *gtlabels[yn];
				if(-1 == gtInd) break;
				Lc += lossMatrix[gtInd][c];
			}
			int ind = firstInd;
			int n = 0;
			while (n < NNEIGHBOURS and ind < NCLASSES) // ?���z?��?C��labels?�s�z?�C�R�@?��NNEIGHBOURS����??�o
			{ 
				// caculate Ly
				for(int yn=0; yn < MAXLABELS; yn++)
				{
					gtInd = *gtlabels[yn];
					if(-1 == gtInd) break;
					Ly += lossMatrix[gtInd][pa[ind]];
				}
				if (Ly >= Lc)
				{
				  if (n == 0) firstInd = ind;
				  witnesses[n++] = ind;
				}
				ind ++;
			}
			if (n < NNEIGHBOURS) break;
			int minLoss = 10000;
			sumPhi = 0;
			for (int n = 0; n < NNEIGHBOURS; n ++)
			{ // calculate part2 in Eq.18
				sumPhi += innerProducts[witnesses[n]];
				int clazz = pa[witnesses[n]]; 
				// calculate the part1 in Eq.18
				for(int yn=0; yn < MAXLABELS; yn++)
				{
					gtInd = *gtlabels[yn];
					if(-1 == gtInd) break;
					minLoss += lossMatrix[gtInd][clazz];
				}
			}
			Ly = minLoss;
			// finally we allocate the predicted Y^n to classes
			if (Ly + sumPhi > maxVal)
			{
				maxVal = Ly + sumPhi;
				for (int n = 0; n < NNEIGHBOURS; n ++)
				  classes[n] = pa[witnesses[n]]; 
			}
		}
		delete [] witnesses;
	}
	
	if (!prePa) delete [] pa;
	if (!preProduct) delete [] innerProducts;
	delete [] phi;
}