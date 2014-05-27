#ifndef _GRAPHDATA_CPP_
#define _GRAPHDATA_CPP_

#include <omp.h>

#include "common.hpp"
#include "sml.hpp"
#include "configuration.hpp"
#include "bmrmexception.hpp"
#include "timer.hpp"
#include "graphdata.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "string.h"

#include "model.hpp"
#include "graphmatchloss.hpp"

#include "mkl.h"
#include <boost/numeric/ublas/matrix.hpp>

#include "cublas.h"
using namespace std;

enum { TRAINING, VALIDATION, TESTING };

void read_libsvm(std::istream& f, float* featVect)
{
	int i;
	char c;
	float x;
	for(;;)
	{
		c = f.get();
		if (!f.good() || c=='\n' ) // �������У�������ǰ�����ͼƬ��ȡ��ɣ��˳�ѭ��
			break;
		if (::isspace(c)) // ����ȡ��c�ǲ��ǿո� �ǿո������һ��index:value
			continue;

		f.unget();
		f>>i;
		i--; //�����ȡ����index�� ��1��ʾ�����0��ʼ����
		if (f.get() != ':')
		{
			f.unget();
			break;
		}

		f>>x; // �����ȡ��valueֵ
		if (!f.good())
			break;

		// nodes.push_back(svm_node(i,x));
		// ������Ҫ��ÿ�ζ�ȡ��index:value�� ����imagefeature�����Ӧ��λ��
		featVect[i] = x;
    }

}

void readFeature(char* filename, float** dataFeat, int& total, int** dataLabels){
	std::cout<<"Loading data\n";
	std::ifstream fin(filename);
	if(!fin.good())
	{
		std::cout<<"Couldn't find file.\n";
		exit(0);
	}

	//First find out the number of labels
	char c;
	int label;
	int num_Images = 0;
	int idx_label = 0;

	d=0;
	fin.clear();
	fin.seekg(0, std::ios::beg);
	int count=0;
	fin.get();

	while(fin.good())
	{
		idx_label=0;
		fin.unget();
		//FullVector y;
		c=fin.get();
		if(c!=' ') // ���if������ڶ�ȡÿһ���еĿ�ʼ�Ķ��label, ֱ������' '����
		{
			fin.unget();
			do
			{
				fin>>label;
				dataLabels[num_Images][idx_label] = label; //��ÿ�ζ�ȡ��label����ֵ���뵽dataLabelsָ����λ��
				num_label++;
			}while(fin.get()==',');

		}
		// Ȼ��ʼ��ȡÿһ�п�ʼ��index:value����������
		float* datum = new float [FEATURESIZE]; // ����һ��vector���ڴ洢D-dimά��feature
		read_libsvm(fin, datum); // ��ȡ������ֵ��datum��ָ����λ��

		// Ȼ��datum���뵽����imageMatrixָ����λ��
        for (int f = 0; f < NFEATURES; f ++)
        	dataFeat[total][f] = datum[f]; // TODO: memcpy
        delete [] datum;

        num_Images++;
		count++;
		fin.get();
	}
	assert(num_Images == Num_Train);
	std::cout<<"Data loaded\n";
	std::cout<<Num_Train<<" examples, "<<Num_Label<<" labels, "<<NFEATURES<<" features\n";
}

float** newMatrix(int x, int y)
{
  float* matflat = new float [x*y];
  float** matsquare = new float* [x];
  for (int i = 0; i < x; i ++)
    matsquare[i] = matflat + i*y;

  return matsquare;
}

CGraphData::CGraphData()
{
  Configuration &config = Configuration::GetInstance();
  imageMatrix = NULL;
  FEATURESIZE = config.GetInt("Model.featDim") + 1;

  // Read the image features for training/testing
  if (config.GetString("Data.inputPrefix") == "desc_whole")
  {
    // 	We do two passes of the file, first to count the entries, and then to initialise the matrix.
//    char* filename = new char [100];
//	sprintf(filename, "/media/disk-1/julian_data/features/desc_whole_%d_dense_%d.fvecs", NFEATURES-1, i);

    // ��ȡѵ��������
    char *filename = "corel5k_train_multilabel.svm";
    // ��ʼ��imageMatrix��imageMultiLabels
    imageMatrix = newMatrix(Num_Train, NFEATURES);
    imageMatrix = newMatrix(Num_Train, MAXLABELS);
    if (!config.GetBool("Prediction.outputFvalAndLabels"))
		  readFeature(filename, imageMatrix, imageMultiLabels);
  }

  // Read the cost matrix
  FILE* f = fopen("/media/disk-1/julian_data/features/cost_matrix.txt", "r");
  if (!f)
  {
    printf("Couldn't read /media/disk-1/julian_data/features/cost_matrix.txt\n");
    exit(1);
  }
  //NOTE: using a constant number of classes (1000) for this dataset. This way the whole matrix gets read even if NCLASSES is set to use only a fraction of the classes.
  lossMatrix = new int* [1000]; // lossMatrix��һ��1000x1000�ľ��� ������d(y, y^n)��ֵ
  for (int i = 0; i < 1000; i ++)
  {
    lossMatrix[i] = new int [1000];
    for (int j = 0; j < 1000; j ++)
    {
      float dat;
      int retval;
      retval = fscanf(f, "%f", &dat);
      assert(retval == 1);
      lossMatrix[i][j] = int(dat + 0.5); // Always integer valued for this dataset.
    }
  }
  fclose(f);
  printf("read cost matrix\n");

  // Create a random indicator vector
  //�����indicator vector��ָ��ʼ��\theta_binarys,
  // ����ͨ��one-verus-allѵ�����ɵ�ÿ��label��Ӧ��wֵ��ά��ΪNFEATURES
  srand(0);
  indicator = new float* [NCLASSES];
  for (int i = 0; i < NCLASSES; i ++)
  {
    indicator[i] = new float [NFEATURES];
    
    char* fname = new char [50];
    sprintf(fname, "/media/disk-1/julian_data/svm/svm_%d_%d.svm", NFEATURES-1, i);
    FILE* f = fopen(fname, "r");
    if (!f)
    {
      printf("Couldn't read %s\n", fname);
      exit(1);
    }
    
    for (int j = 0; j < NFEATURES; j ++)
    {
      float feat = 0;
      int retval;
      retval = fscanf(f, "%f", &feat);
      assert(retval == 1);
      indicator[i][j] = feat;
    }
    fclose(f);
    delete [] fname;
  }
  printf("read indicator vectors\n");
  
  latentLabels = NULL;
  if (!config.GetBool("Prediction.outputFvalAndLabels"))
  {
    CModel previousModel;
    Scalar* w = NULL;
    try
    {
      previousModel.Initialize(config.GetString("Model.hotStartModel"));
      w = previousModel.GetW().Data();
    }
    catch (CBMRMException e)
    {
      cout << "No such model: " << config.GetString("Model.hotStartModel") << endl;
    }

    latentLabels = new int* [_N];
    #pragma omp parallel for
    for (int i = 0; i < _N; i ++)
    {
      int* classes = new int [NNEIGHBOURS];
      int* latent = new int [NNEIGHBOURS - 1];
      int label = imageLabels[i];
      assignment(imageMatrix[i], w, indicator, classes, 0, label, lossMatrix, NULL, NULL);
      int ind = 0;
      for (int n = 0; n < NNEIGHBOURS; n ++)
      {
        if (ind == NNEIGHBOURS - 1) break;
        if (classes[n] == label) continue; //classesΪZ_n��ȡֵ�� ���Z_n��y_n��ͬ��������
        latent[ind ++] = classes[n];
      }
      latentLabels[i] = latent;
      delete [] classes;
    }
    printf("initialized latent variables\n"); 
  }
  
  printf("Copying feature matrix to GPU\n");
  cublasInit();
  if (cublasAlloc(NFEATURES*_N, sizeof(**imageMatrix), (void**)&devPtrF) != CUBLAS_STATUS_SUCCESS) { printf("Failed to allocate matrix F\n"); exit(1); }
  if (cublasSetMatrix(NFEATURES, _N, sizeof(**imageMatrix), *imageMatrix, NFEATURES, devPtrF, NFEATURES) != CUBLAS_STATUS_SUCCESS) { printf("Failed to copy matrix F\n"); exit(1); }
  printf("Finished loading data\n");
}

CGraphData::~CGraphData()
{
  delete [] *imageMatrix; // Stored as a flat array.
  delete [] imageMatrix;

  for (int i = 0; i < 1000; i ++)
    delete [] lossMatrix [i];
  delete [] lossMatrix;

  for (int i = 0; i < NCLASSES; i ++)
    delete [] indicator[i];
  delete [] indicator;
  
  if (latentLabels)
  {
    for (int i = 0; i < _N; i ++)
      delete [] latentLabels[i];
    delete [] latentLabels;
  }
  
  cublasFree(devPtrF);
  cublasShutdown();
}

#endif
