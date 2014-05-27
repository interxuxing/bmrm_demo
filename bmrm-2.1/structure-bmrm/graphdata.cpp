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
		if (!f.good() || c=='\n' ) // 遇到换行，表明当前的这个图片读取完成，退出循环
			break;
		if (::isspace(c)) // 看读取的c是不是空格， 是空格就是下一个index:value
			continue;

		f.unget();
		f>>i;
		i--; //这里读取的是index， 减1表示数组从0开始索引
		if (f.get() != ':')
		{
			f.unget();
			break;
		}

		f>>x; // 这里读取的value值
		if (!f.good())
			break;

		// nodes.push_back(svm_node(i,x));
		// 这里需要将每次读取的index:value对 放入imagefeature数组对应的位置
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
		if(c!=' ') // 这个if语句用于读取每一行中的开始的多个label, 直到遇到' '结束
		{
			fin.unget();
			do
			{
				fin>>label;
				dataLabels[num_Images][idx_label] = label; //将每次读取的label索引值放入到dataLabels指定的位置
				num_label++;
			}while(fin.get()==',');

		}
		// 然后开始读取每一行开始的index:value的特征数据
		float* datum = new float [FEATURESIZE]; // 创建一个vector用于存储D-dim维的feature
		read_libsvm(fin, datum); // 读取后续的值到datum的指定的位置

		// 然后将datum放入到整个imageMatrix指定的位置
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

    // 读取训练集数据
    char *filename = "corel5k_train_multilabel.svm";
    // 初始化imageMatrix和imageMultiLabels
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
  lossMatrix = new int* [1000]; // lossMatrix是一个1000x1000的矩阵， 用来存d(y, y^n)的值
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
  //这里的indicator vector是指初始的\theta_binarys,
  // 就是通过one-verus-all训练生成的每个label对应的w值，维度为NFEATURES
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
        if (classes[n] == label) continue; //classes为Z_n的取值， 如果Z_n与y_n相同，则跳过
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
