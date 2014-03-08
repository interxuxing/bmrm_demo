/* Copyright (c) 2011, (removed for blind review)
 * All rights reserved. 
 * 
 * The contents of this file are subject to the Mozilla Public License 
 * Version 1.1 (the "License"); you may not use this file except in 
 * compliance with the License. You may obtain a copy of the License at 
 * http://www.mozilla.org/MPL/ 
 * 
 * Software distributed under the License is distributed on an "AS IS" 
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the 
 * License for the specific language governing rights and limitations 
 * under the License. 
 * 
 * Author: removed for blind review
 */

#include "common.hpp"
#include "sml.hpp"
#include "configuration.hpp"
#include "bmrmexception.hpp"
#include "timer.hpp"
#include "genericdata.hpp"
#include <vector>
#include <math.h>
#include "preader.h"
#include "words.h"
#include "mt.h"
#include "set"
#include "model.hpp"

using namespace std;

// D dimensional inner product
Cost inner_product(double* x, double* y, int D)
{
  double res = 0;
  for (int i = 0; i < D; i ++)
    res += (Cost) x[i] * (Cost) y[i];
  return res;
}

CGenericData::CGenericData()
{
  Configuration &config = Configuration::GetInstance();
  
  string nodeFile = config.GetString("Data.nodeFeaturesTrain");
  string evidenceFile = nodeFile;
  string edgeFile = config.GetString("Data.edgeFeaturesTrain");
  string imageFeatures = config.GetString("Data.imageFeaturesTrain");
  string textFile = config.GetString("Data.textFeaturesTrain");
  if (config.GetBool("Prediction.outputFvalAndLabels"))
  {
    nodeFile = config.GetString("Data.nodeFeaturesTest");
    edgeFile = config.GetString("Data.edgeFeaturesTest");
    imageFeatures = config.GetString("Data.imageFeaturesTest");
    textFile = config.GetString("Data.textFeaturesTest");
  }
  learnLabel = config.GetInt("Data.learnLabel");
  string idFile = config.GetString("Data.idFile");
  string textIdFile = config.GetString("Data.textIdFile");

  // Read image features  
  FILE* imf = NULL;
  int nImageFeatures = 0;
  if (config.GetBool("Data.useImageFeatures"))
  {
    imf = fopen(imageFeatures.c_str(), "rb");
    fseek(imf, 28, SEEK_CUR);
    fread(&nImageFeatures, 4, 1, imf);
    fclose(imf);

    imf = fopen(imageFeatures.c_str(), "r");
    char c;
    while ((c = fgetc(imf)) != '\n')
      if (c == ' ') nImageFeatures ++;
    fclose(imf);
  }
  
  // Read group and tag features
  FILE* idf = fopen(idFile.c_str(), "r");
  int currentID;
  vector<int> groupsToUse;
  vector<int> tagsToUse;
  set<int> groupsToUseSet;
  set<int> tagsToUseSet;
  int NGroups_;
  int NTags_;
  while (fscanf(idf, "%d", &currentID) == 1)
  {
    char* name = new char [1000];
    fscanf(idf, "%s %d %d", name, &NGroups_, &NTags_);
    for (int i = 0; i < NGroups_; i ++)
    {
      int x;
      float f;
      fscanf(idf, "%d:%f", &x, &f);
      if (currentID == learnLabel)
        if (f > 2)
        {
          groupsToUse.push_back(x);
          groupsToUseSet.insert(x);
        }
    }
    for (int i = 0; i < NTags_; i ++)
    {
      int x;
      float f;
      fscanf(idf, "%d:%f", &x, &f);
      if (currentID == learnLabel)
        if (f > 2 and not (config.GetBool("Data.baseline")))
        {
          tagsToUse.push_back(x);
          tagsToUseSet.insert(x);
        }
    }
    delete [] name;
    if (currentID == learnLabel) break;
  }
  fclose(idf);
  
  // Features for flat model (only used by 'flat' baseline)
  int NOthers = 0;
  FILE* flatFile = NULL;
  if (config.IsSet("Data.flatFeaturesTrain") or config.IsSet("Data.flatFeaturesTest"))
  {
    if (config.GetBool("Prediction.outputFvalAndLabels"))
      flatFile = fopen(config.GetString("Data.flatFeaturesTest").c_str(), "r");
    else
      flatFile = fopen(config.GetString("Data.flatFeaturesTrain").c_str(), "r");
    if (flatFile == NULL)
    {
      printf("Problem reading features\n");
      exit(1);
    }
    fscanf(flatFile, "%d", &NOthers);
  }
  
  // Text features
  FILE* tidf = fopen(textIdFile.c_str(), "r");
  vector<int> wordsToUse;
  map<int, int> wordsToUseMap;
  int NWords_;
  int wordId = 0;
  while (fscanf(tidf, "%d", &currentID) == 1)
  {
    char* name = new char [1000];
    fscanf(tidf, "%s %d", name, &NWords_);
    for (int i = 0; i < NWords_; i ++)
    {
      int x;
      float f;
      fscanf(tidf, "%d:%f", &x, &f);
      if (currentID == learnLabel)
      {
        if (f > 2 and not (config.GetBool("Data.baseline")))
        {
          wordsToUse.push_back(x);
          wordsToUseMap[x] = wordId ++;
        }
      }
    }
    delete [] name;
    if (currentID == learnLabel) break;
  }
  fclose(tidf);
  
  // Use only the first 1000 most popular words
  for (int i = 0; i < 1000; i ++)
  {
    if (wordsToUseMap.find(i) == wordsToUseMap.end() and not (config.GetBool("Data.baseline")))
    {
      wordsToUse.push_back(i);
      wordsToUseMap[i] = wordId ++;
    }
  }
  
  int nGroupsPositive = groupsToUse.size();
  int nTagsPositive = tagsToUse.size();
  
  // Read remaining node features
  FILE* nf = fopen(nodeFile.c_str(), "r");

  fscanf(nf, "%d %d %d", &NGroups, &NTags, &NLabels);
  for (int i = 0; i < NGroups; i ++)
  {
    int id = 0;
    char* name = new char [1000];
    fscanf(nf, "%d %s", &id, name);
    labelNames[id] = string(name);
    if (i < 1000 and groupsToUseSet.find(id) == groupsToUseSet.end()) { groupsToUse.push_back(id); }
    delete [] name;
  }
  for (int i = 0; i < NTags; i ++)
  {
    int id = 0;
    char* name = new char [1000];
    fscanf(nf, "%d %s", &id, name);
    labelNames[id] = string(name);
    if (i < 1000 and tagsToUseSet.find(id) == tagsToUseSet.end()) { tagsToUse.push_back(id); }
    delete [] name;
  }
  for (int i = 0; i < NLabels; i ++)
  {
    int id = 0;
    char* name = new char [1000];
    fscanf(nf, "%d %s", &id, name);
    labelNames[id] = string(name);
    delete [] name;
  }
  fscanf(nf, "%d", &NPhotos);
  
  NGroups_ = groupsToUse.size();
  NTags_ = tagsToUse.size();
  NWords_ = wordsToUse.size();

  if (!config.GetBool("Data.useGroupFeatures"))
  {
    NGroups_= 0;
  }
  if (!config.GetBool("Data.useTagFeatures"))
  {
    NTags_ = 0;
  }
  
  if (config.GetBool("Data.baseline"))
  {
    NGroups_ = 0;
    NWords_ = 0;
  }

  if (config.GetBool("Data.useNodeFeatures"))
    nNodeFeatures = nImageFeatures + NGroups_ + NTags_ + NWords_ + NOthers;
  else
  {
    nNodeFeatures = nImageFeatures;
    NGroups_ = 0;
    NTags_ = 0;
    NWords_ = 0;
  }

  // Scale features
  lossPositive = 0;
  lossNegative = 0;
  for (int i = 0; i < NPhotos; i ++)
  {
    long photoId = 0;
    char* userId = new char [50];
    char* indicator = new char [NGroups + NTags + NLabels + 2];
    fscanf(nf, "%ld %s %s", &photoId, userId, indicator);

    nodeIndex[photoId] = i;
    indexNode[i] = photoId;

    double* feature = new double [nNodeFeatures];
    for (int f = 0; f < nNodeFeatures; f ++)
      feature[f] = 0;
    if (config.GetBool("Data.useNodeFeatures"))
    {
      for (int i = 0; i < NGroups_; i ++)
      {
        if (indicator[groupsToUse[i]] == '1')
          feature[nImageFeatures + i] = 1.0/NGroups_;
        else feature[nImageFeatures + i] = 0;
      }
      for (int i = 0; i < NTags_; i ++)
      {
        if (indicator[tagsToUse[i]] == '1')
          feature[nImageFeatures + NGroups_ + i] = 1.0/NTags_;
        else feature[nImageFeatures + NGroups_ + i] = 0;
      }
    }

    nodeFeatures[i] = feature;
    nodeLabels[i] = NEGATIVE;
    if (indicator[learnLabel] == '1') nodeLabels[i] = POSITIVE;
    if (indicator[learnLabel] == '0') nodeLabels[i] = NEGATIVE;
    if (nodeLabels[i] == POSITIVE) lossPositive ++;
    if (nodeLabels[i] == NEGATIVE) lossNegative ++;

    delete [] userId;
    delete [] indicator;
  }
  fclose(nf);

  if (config.GetBool("Data.useNodeFeatures"))
  {
    FILE* tf = fopen(textFile.c_str(), "r");
    int NWords;
    fscanf(tf, "%d", &NWords);
    for (int i = 0; i < NWords; i ++)
    {
      int wordID;
      char* word = new char [1000];
      fscanf(tf, "%d %s", &wordID, word);
      delete [] word;
    }

    for (int i = 0; i < NPhotos; i ++)
    {
      long photoId = 0;
      char* userId = new char [1000];
      int nf;
      if (not fscanf(tf, "%ld %s %d", &photoId, userId, &nf))
      {
        printf("Expected ID and int\n");
        exit(1);
      }

      double* feature = nodeFeatures[nodeIndex[photoId]] + nImageFeatures + NGroups_ + NTags_;
      for (int i = 0; i < NWords_; i ++)
      {
        feature[i] = 0;
      }
      for (int i = 0; i < nf; i ++)
      {
        int hkey, hvalue;
        fscanf(tf, "%d:%d", &hkey, &hvalue);
        if (wordsToUseMap.find(hkey) != wordsToUseMap.end())
          feature[wordsToUseMap[hkey]] = hvalue * 1.0/NWords_;
      }
      delete [] userId;
    }
    fclose(tf);
  }
  
  if (flatFile)
  {
    for (int i = 0; i < NPhotos; i ++)
    {
      long photoId = 0;
      char* indicator = new char [NOthers + 1];
      if (not fscanf(flatFile, "%ld %s", &photoId, indicator))
      {
        printf("Expected ID and indicator\n");
        exit(1);
      }
      double* features = nodeFeatures[nodeIndex[photoId]] + nImageFeatures + NGroups_ + NTags_ + NWords_;
      for (int f = 0; f < NOthers; f ++)
      {
        features[f] = indicator[f] == '1' ? 1 : 0;
      }
    }
  }

  if (config.GetBool("Data.trainingEvidence") and config.GetBool("Prediction.outputFvalAndLabels"))
  {
    nf = fopen(evidenceFile.c_str(), "r");
    int NGroups_;
    int NTags_;
    int NLabels_;

    fscanf(nf, "%d %d %d", &NGroups_, &NTags_, &NLabels_);
    for (int i = 0; i < NGroups_; i ++)
    {
      int id = 0;
      char* name = new char [1000];
      fscanf(nf, "%d %s", &id, name);
      delete [] name;
    }
    for (int i = 0; i < NTags_; i ++)
    {
      int id = 0;
      char* name = new char [1000];
      fscanf(nf, "%d %s", &id, name);
      delete [] name;
    }
    for (int i = 0; i < NLabels_; i ++)
    {
      int id = 0;
      char* name = new char [1000];
      fscanf(nf, "%d %s", &id, name);
      delete [] name;
    }
    int NPhotos_;
    fscanf(nf, "%d", &NPhotos_);

    for (int i = 0; i < NPhotos_; i ++)
    {
      long photoId = 0;
      char* userId = new char [50];
      char* indicator = new char [NGroups_ + NTags_ + NLabels_ + 1];
      fscanf(nf, "%ld %s %s", &photoId, userId, indicator);
      
      int j = nodeIndex[photoId];
      if (indicator[learnLabel] == '1')
      {
        if (nodeLabels.find(j) != nodeLabels.end() and nodeLabels[j] == POSITIVE) lossPositive --;
        nodeLabels[j] = EVIDENCE_POSITIVE;
      }
      if (indicator[learnLabel] == '0')
      {
        if (nodeLabels.find(j) != nodeLabels.end() and nodeLabels[j] == NEGATIVE) lossNegative --;
        nodeLabels[j] = EVIDENCE_NEGATIVE;
      }

      delete [] userId;
      delete [] indicator;
    }
    fclose(nf);
  }
  printf("Read node features\n");

  if (config.GetBool("Data.useImageFeatures"))
  {
    imf = fopen(imageFeatures.c_str(), "rb");
    while (!feof(imf))
    {
      long photoID = 0;
      int elementsPerPoint = 0;
      int dimensionCount = 0;
      int pointCount = 0;
      int bytesPerElement = 0;
      fread(&photoID, 8, 1, imf);
      if (feof(imf)) break;
      fseek(imf, 16, SEEK_CUR);
      fread(&elementsPerPoint, 4, 1, imf);
      fread(&dimensionCount, 4, 1, imf);
      fread(&pointCount, 4, 1, imf);
      fread(&bytesPerElement, 4, 1, imf);
      fseek(imf, (elementsPerPoint * pointCount * bytesPerElement), SEEK_CUR);
      if (dimensionCount != nImageFeatures)
      {
        printf("Expected %d features but got %d\n", nImageFeatures, dimensionCount);
        exit(1);
      }
      fread(nodeFeatures[nodeIndex[photoID]], bytesPerElement, dimensionCount, imf);
    }
    fclose(imf);
    
    imf = fopen(imageFeatures.c_str(), "r");
    long photoID = 0;
    float f;
    while (fscanf(imf, "%ld", &photoID) == 1)
    {
      for (int i = 0; i < nImageFeatures; i ++)
      {
        fscanf(imf, "%f", &f);
        nodeFeatures[nodeIndex[photoID]][i] = f / nImageFeatures;
      }
    }
    fclose(imf);
    
    printf("Read image features\n");
  }

  int useGroupFeatures = config.GetBool("Data.useGroupFeatures");
  int useTagFeatures = config.GetBool("Data.useTagFeatures");
  if (config.GetBool("Data.useSocialFeatures"))
  {
    FILE* ef = fopen(edgeFile.c_str(), "r");
    int NE_;
    fscanf(ef, "%d %d", &NE_, &nEdgeFeatures);
    
    NE = 0;
    int thisNode = 1;
    long prevID = -1;
    for (int e = 0; e < NE_; e ++)
    {
      long nID1 = 0; long nID2 = 0;
      fscanf(ef, "%ld %ld", &nID1, &nID2);
      if (nID1 == prevID)
        thisNode ++;
      else
        thisNode = 1;
      prevID = nID1;
      
      double* feature = new double [nEdgeFeatures];
      
      int total = 0;
      for (int j = 0; j < nEdgeFeatures; j ++)
      {
        float feat = 0;
        fscanf(ef, "%f", &feat);
        feature[j] = feat;
        if (j >= 2) total += feature[j] > 0 ? 1 : 0;
      }
      if (nID1 == nID2)
      {
        printf("%ld %ld\n", nID1, nID2);
        printf("Got self loop\n");
        exit(1);
      }
      
      if (!useGroupFeatures) feature[0] = 0;
      if (!useTagFeatures) feature[1] = 0;
      
      if (thisNode <= 100) // Don't construct more than 100 edges per node
      {
        edgeIndex[pair<int,int>(nodeIndex[nID1],nodeIndex[nID2])] = NE;
        indexEdge[NE] = pair<int,int>(nodeIndex[nID1],nodeIndex[nID2]);
        edgeFeatures[NE] = feature;
        NE ++;
      }
      else
        delete [] feature;
    }
    fclose(ef);
    printf("Read edge features\n");
  }
  else
  {
    NE = 0;
    nEdgeFeatures = 0;
  }
  
  firstOrderResponses = NULL;
  if (config.GetBool("Data.useSocialFeatures") and config.IsSet("Data.firstOrderModel"))
  {
    firstOrderResponses = new map<int, double>();
    string firstOrderFile = config.GetString("Data.firstOrderModel");
    FILE* m = fopen(firstOrderFile.c_str(), "r");
    while (fgetc(m) != '\n');
    while (fgetc(m) != '\n');
    double* w = new double [nNodeFeatures];
    for (int i = 0; i < nNodeFeatures; i ++)
    {
      float f;
      fscanf(m, "%f", &f);
      w[i] = f;
    }
    for (map<int, double*>::iterator it = nodeFeatures.begin(); it != nodeFeatures.end(); it ++)
    {
      (*firstOrderResponses)[it->first] = inner_product(it->second, w, nNodeFeatures);
    }
    
    delete [] w;
  }
  
  if (lossPositive < 0.5)
  {
    printf("Not enough positive examples!\n");
    exit(1);
  }

  num_positive = nEdgeFeatures;

  N = 1;
}

CGenericData::~CGenericData()
{
  for (map<int, double*>::iterator it = nodeFeatures.begin(); it != nodeFeatures.end(); it ++)
    delete [] it->second;
  for (map<int, double*>::iterator it = edgeFeatures.begin(); it != edgeFeatures.end(); it ++)
    delete [] it->second;
  
  delete firstOrderResponses;
}

