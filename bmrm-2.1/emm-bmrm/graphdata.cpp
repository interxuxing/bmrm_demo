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

int FEATURESIZE;

// Adapted from the INRIA MATLAB file at:
// http://pascal.inrialpes.fr/data/holidays/fvecs_read.m
void readFVECS(char* filename,
               float** dataM,
               int& total,
               std::vector<int>& labels,
               int label,
               int mode,
               map<string,int>* classMap,
               FILE* classFile)
{
  FILE* f = fopen(filename, "rb");
  if (f == NULL)
  {
    printf("Couldn't read %s\n", filename);
    exit(1);
  }

  // Read the feature vectors from the file
  int count = 0;
  int size = 0;
  int retval;
  while ((retval = fread(&size, sizeof(int), 1, f)) == 1)
  {
    if (size != FEATURESIZE - 1)
    {
      printf("Expected %d dimensional features, got %d dimensional features\n", FEATURESIZE - 1, size);
      exit(1);
    }
    float* datum = new float [FEATURESIZE];
    Scalar featnorm = 0;
    for (int i = 0; i < size; i ++)
    {
      float dat;
      retval = fread(&dat, sizeof(float), 1, f);
      featnorm += fabs(dat);
      datum[i] = float(dat);
    }
    for (int i = 0; i < size; i ++)
      datum[i] = sqrt(datum[i])/sqrt(featnorm); // here normalized to unit vector
    datum[FEATURESIZE - 1] = 1; // Bias parameter

    // Use 'mode' to read only a fraction of the file. It should determine which part of the file is used for:
    // training
    // validation
    // testing
    if (mode == TESTING)
    {
      char* name = new char [50];
      retval = fscanf(classFile, "%s", name);
      name[9] = '\0';
      string sName(name);
      for (int f = 0; f < NFEATURES; f ++)
        dataM[total][f] = datum[f];
      labels.push_back(classMap->at(sName));
      delete [] name;
      total ++;
    }
    else if (count % FRACTION == mode and mode != TESTING)
    {
      if (dataM)
      {
        for (int f = 0; f < NFEATURES; f ++)
          dataM[total][f] = datum[f]; // TODO: memcpy

        // In the training files, each file contains instances from a single class.
        labels.push_back(label);
      }
      total ++;
    }

    delete [] datum;
    count ++;
  }
  fclose(f);

  if (count == 0)
  {
    printf("Failed to read any data from %s\n", filename);
    exit(1);
  }
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
    for (int pass = 0; pass < 2; pass ++)
    { // We do two passes of the file, first to count the entries, and then to initialise the matrix.
      int total = 0;
      for (int i = 0; i < NCLASSES; i ++)
      {
        char* filename = new char [100];
        sprintf(filename, "/media/disk-1/julian_data/features/desc_whole_%d_dense_%d.fvecs", NFEATURES-1, i);

        if (!config.GetBool("Prediction.outputFvalAndLabels"))
          readFVECS(filename, imageMatrix, total, imageLabels, i, TRAINING, NULL, NULL);
        else
          readFVECS(filename, imageMatrix, total, imageLabels, i, VALIDATION, NULL, NULL);
        delete [] filename;
      }
      if (pass == 0)
      {
        _N = total;
        imageMatrix = newMatrix(_N, NFEATURES);
      }
    }
  }
  else
  {
    FILE* f = fopen("/media/disk-1/julian_data/features/changes_class_imagenetTrain", "r");
    map<string, int> className;
    char* name = new char [50];
    int ind = 0;
    int label = 0;
    int retval;
    while ((retval = fscanf(f, "%d %s", &ind, name)) == 2)
    {
      string cName(name);
      className[cName] = label ++;
    }
    delete [] name;
    fclose(f);
    f = fopen("/media/disk-1/julian_data/features/test.list", "r");
    _N = 150000;
    int total = 0;
    imageMatrix = newMatrix(_N, NFEATURES);

    for (int i = 0; i < 150; i ++)
    {
      char* filename = new char [100];
      sprintf(filename, "/media/disk-1/julian_data/features/desc_test_%d_dense_%d.fvecs", NFEATURES-1, i);

      if (!config.GetBool("Prediction.outputFvalAndLabels"))
      {
        printf("Don't do learning on the test data\n");
        exit(1);
      }
      else
        readFVECS(filename, imageMatrix, total, imageLabels, -1, TESTING, &className, f);
      delete [] filename;
    }
    fclose(f);
  }
  printf("read training/test data (N = %d)\n", _N);

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

  // Create a random indicator vector.
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
  //这里的indicator vector是指初始的\theta_binarys, 就是通过one-verus-all训练生成的每个label对应的w值，维度为NFEATURES
  
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
