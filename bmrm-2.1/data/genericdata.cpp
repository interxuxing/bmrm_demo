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
  
  string nodeFile = config.GetString("Data.nodeFeaturesTrain"); // data/trainingIndicatorsPASCAL.txt
  string evidenceFile = nodeFile; // data/trainingIndicatorsPASCAL.txt
  string edgeFile = config.GetString("Data.edgeFeaturesTrain");  // data/trainingEdgeFeaturesPASCAL.txt
  string imageFeatures = config.GetString("Data.imageFeaturesTrain"); // null
  string textFile = config.GetString("Data.textFeaturesTrain"); // data/trainingTextPASCAL.txt
  if (config.GetBool("Prediction.outputFvalAndLabels"))  // null, ����������test����������֤��, ԭʼ��conf�ļ���������train+test
  {
    nodeFile = config.GetString("Data.nodeFeaturesTest");
    edgeFile = config.GetString("Data.edgeFeaturesTest");
    imageFeatures = config.GetString("Data.imageFeaturesTest");
    textFile = config.GetString("Data.textFeaturesTest");
  }
  learnLabel = config.GetInt("Data.learnLabel"); // 6043
  printf("Now load feature datas for Label %d \n", learnLabel);
  string idFile = config.GetString("Data.idFile"); // data/trainingGroupIDsPASCAL.txt
  string textIdFile = config.GetString("Data.textIdFile"); // data/trainingWordIDsPASCAL.txt

  // Read image features  
  FILE* imf = NULL;
  int nImageFeatures = 0;
  if (config.GetBool("Data.useImageFeatures")) // false, demo������û���õ�imagefeature
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
  // idFile =  data/trainingGroupIDsPASCAL.txt, ����ļ����Բ��øģ� ��Ϊ�������˼���dataset��group��Ϣ
  FILE* idf = fopen(idFile.c_str(), "r");
  int currentID;
  vector<int> groupsToUse;
  vector<int> tagsToUse;
  set<int> groupsToUseSet;
  set<int> tagsToUseSet;
  int NGroups_;
  int NTags_;
  while (fscanf(idf, "%d", &currentID) == 1) // ÿ��ѭ��ȡһ��currentID, ÿһ�е�һ��ֵ�������һ�е� 26041
  {
    char* name = new char [1000];
    fscanf(idf, "%s %d %d", name, &NGroups_, &NTags_);
    // ���� PASCAL:aeroplane 200 0 -> name=PASCAL:aeroplane, NGroups_=200 NTags_=0
    // ÿһ�к����ɶԵ�����x:f ������NGroups_����NTags_��˳����
    for (int i = 0; i < NGroups_; i ++)
    {
      int x;
      float f;
      fscanf(idf, "%d:%f", &x, &f);
      if (currentID == learnLabel) // learnLabel=6043Ϊaeroplane, ������ֵ�� ����ѡ��ͬ������
        if (f > 2) // f����2�Ŵ�
        {
          groupsToUse.push_back(x); // vector��push_back
          groupsToUseSet.insert(x); // set��insert
        }
    }
    for (int i = 0; i < NTags_; i ++)
    {
      int x;
      float f;
      fscanf(idf, "%d:%f", &x, &f);
      if (currentID == learnLabel)
        if (f > 2 and not (config.GetBool("Data.baseline"))) // Data.baseline = false
        {
          tagsToUse.push_back(x);
          tagsToUseSet.insert(x);
        }
    }
    delete [] name;
    if (currentID == learnLabel) break; // ֻѡlearnLabel��һ��
  }
  fclose(idf);
  
  // Features for flat model (only used by 'flat' baseline)
  int NOthers = 0;
  FILE* flatFile = NULL;
  if (config.IsSet("Data.flatFeaturesTrain") or config.IsSet("Data.flatFeaturesTest")) // Data.flatFeaturesTrain = Data.flatFeaturesTest = false
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
  // textIdFile = data/trainingWordIDsPASCAL.txt, ����ļ����øģ� �������˼���dataset��word��Ϣ
  FILE* tidf = fopen(textIdFile.c_str(), "r");
  vector<int> wordsToUse;
  map<int, int> wordsToUseMap;
  int NWords_;
  int wordId = 0;
  while (fscanf(tidf, "%d", &currentID) == 1)
  {
    char* name = new char [1000];
    fscanf(tidf, "%s %d", name, &NWords_); // name=PASCAL:aeroplane, NWwords_=1639
    for (int i = 0; i < NWords_; i ++)
    {
      int x;
      float f;
      fscanf(tidf, "%d:%f", &x, &f); // ���x:f��ʽ��֮ǰ��idf�Ǹ��ļ���groups/tags����
      if (currentID == learnLabel)
      {
        if (f > 2 and not (config.GetBool("Data.baseline"))) //f>2��ָfrequency>2 ?
        {
          wordsToUse.push_back(x);
          wordsToUseMap[x] = wordId ++; // x��word��id�� wordId++�ǰѳ��ִ����ۼӣ������map��<x, wordId>
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
  // nodeFile = data/trainingIndicatorsPASCAL.txt, ����ļ���Ӧ��nodeFeaturesXXX.txt�ļ�
  FILE* nf = fopen(nodeFile.c_str(), "r");
  // nodeFile����ļ��� ��һ�д��������Ϣ�� groups, tags, labels����Ŀ
  fscanf(nf, "%d %d %d", &NGroups, &NTags, &NLabels); // NGroups=6041 NTags=20000 NLabels=215
  // �ӵڶ��п�ʼ�� �ȴ���groups, Ȼ��tags�� ���labels
  // groups��ÿһ�еĸ�ʽΪ: id, name, ����0 20759249@N00
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
  // ǰ����������ngroups, ntags, nlabels��ȡ���Ժ󣬽�����һ��(��6041+20000+215+1��)��8151��Ϊnphotos�ĸ���
  fscanf(nf, "%d", &NPhotos);
  
  NGroups_ = groupsToUse.size();
  NTags_ = tagsToUse.size();
  NWords_ = wordsToUse.size();

  if (!config.GetBool("Data.useGroupFeatures")) // useGroupFeatures=true
  {
    NGroups_= 0; //δִ��
  }
  if (!config.GetBool("Data.useTagFeatures")) // useTagFeatures=false
  {
    NTags_ = 0; // ִ����
  }
  
  if (config.GetBool("Data.baseline")) // baseline=false
  {
    NGroups_ = 0; // δִ��
    NWords_ = 0;
  }

  if (config.GetBool("Data.useNodeFeatures")) // useNodeFeatures=true
    nNodeFeatures = nImageFeatures + NGroups_ + NTags_ + NWords_ + NOthers; // ����weight����������
  else
  {
    nNodeFeatures = nImageFeatures;
    NGroups_ = 0;
    NTags_ = 0;
    NWords_ = 0;
  }

  // ���￪ʼ��ȡnphotos�����features
  // Scale features
  lossPositive = 0;
  lossNegative = 0;
  for (int i = 0; i < NPhotos; i ++)
  {
    long photoId = 0;
    char* userId = new char [50];
    char* indicator = new char [NGroups + NTags + NLabels + 2];
    // ����ÿһ�еĴ�Ÿ�ʽΪ 111604571 52734910@N00 ....���ܳ���indicators, ע��ÿ��photo������NGroups + NTags + NLabels + 2ά�ȵ�indicator��
    fscanf(nf, "%ld %s %s", &photoId, userId, indicator);

    nodeIndex[photoId] = i;
    indexNode[i] = photoId;

    double* feature = new double [nNodeFeatures];
    for (int f = 0; f < nNodeFeatures; f ++)
      feature[f] = 0;
    if (config.GetBool("Data.useNodeFeatures")) // useNodeFeatures=true
    {
      for (int i = 0; i < NGroups_; i ++)
      {
        if (indicator[groupsToUse[i]] == '1')
          feature[nImageFeatures + i] = 1.0/NGroups_; //��һ��
        else feature[nImageFeatures + i] = 0;
      }
      for (int i = 0; i < NTags_; i ++)
      {
        if (indicator[tagsToUse[i]] == '1')
          feature[nImageFeatures + NGroups_ + i] = 1.0/NTags_; //��һ��
        else feature[nImageFeatures + NGroups_ + i] = 0;
      }
    }

    nodeFeatures[i] = feature; // ÿһ��photo��nodeFeatures��ά����[nImageFeatures+NGroups_+NTags_+NWords_+NOthers]
    nodeLabels[i] = NEGATIVE; // nodeLabel��С��[NPhotos], ��ʼ����Ϊ-1�� ���learnLabelָ���� ����Ϊ+1
    if (indicator[learnLabel] == '1') nodeLabels[i] = POSITIVE; //indicator��С��[NGroups+NTags+NLabels+2], ��indicator�Ǹ��ļ��ж�ȡ��
    if (indicator[learnLabel] == '0') nodeLabels[i] = NEGATIVE;
    if (nodeLabels[i] == POSITIVE) lossPositive ++; //ͳ��positive�ĸ���
    if (nodeLabels[i] == NEGATIVE) lossNegative ++; //ͳ��negative�ĸ���

    delete [] userId;
    delete [] indicator;
  }
  printf("For learnLabel: %d, num of positive: %f, negative: %f \n", learnLabel, lossPositive, lossNegative);
  fclose(nf);

  if (config.GetBool("Data.useNodeFeatures")) // useNodeFeatures=true
  {
	  // ��ȡdata/trainingTextPASCAL.txt����ļ�
    FILE* tf = fopen(textFile.c_str(), "r"); // textFile=data/trainingTextPASCAL.txt
    int NWords;
    // �ļ���һ����ͳ��words������NWords=14227
    fscanf(tf, "%d", &NWords);
    // ֮�����ж�ȡ wordID:word�ĸ�ʽ, ����6 shot, ע�������ȡ��wordID:word����û��ʹ�ã�
    for (int i = 0; i < NWords; i ++)
    {
      int wordID;
      char* word = new char [1000];
      fscanf(tf, "%d %s", &wordID, word);
      delete [] word;
    }
    // ����word��ȡ���Ժ� ��������photos�Ĵ洢�� ��ʼһ��(14229��)��ʽΪ8151111604571 52734910@N00 3 ...
    // �����Ӧ��ÿ��photo��userid�ļ�Ĺ�ϵ�� ÿ��wordid���ֵĴ���
    for (int i = 0; i < NPhotos; i ++)
    {
      long photoId = 0;
      char* userId = new char [1000];
      int nf;
      // 8151111604571 52734910@N00 3��ʾ photoID, userID, nf
      if (not fscanf(tf, "%ld %s %d", &photoId, userId, &nf))
      {
        printf("Expected ID and int\n");
        exit(1);
      }
      // feature���ָ�붨λ��photo�Ŀ�ʼ��
      double* feature = nodeFeatures[nodeIndex[photoId]] + nImageFeatures + NGroups_ + NTags_;
      for (int i = 0; i < NWords_; i ++)
      {
        feature[i] = 0;
      }
      // Ȼ���ȡnf��photo feature�� �����14229�У� nf=3 ����674:1 5908:1 63:1�� ��ʾĳ��word���ֵĴ���
      for (int i = 0; i < nf; i ++)
      {
        int hkey, hvalue;
        fscanf(tf, "%d:%d", &hkey, &hvalue);
        if (wordsToUseMap.find(hkey) != wordsToUseMap.end())
          feature[wordsToUseMap[hkey]] = hvalue * 1.0/NWords_; //��һ��
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

  // trainingEvidence=false û��outputFvalAndLabels, ���²�ִ��
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

  // �˶��Ƕ�ȡimagefeature�ļ��ģ� ���Ǵ�demo��useImageFeatures=false, ���³����δִ��
  if (config.GetBool("Data.useImageFeatures"))
  {
	  //�Ӵ˴��������� imageFeatureӦ���Ǵ���ڵ�һһ���ļ��У� ÿһ��Ϊһ��photo��feature
    imf = fopen(imageFeatures.c_str(), "rb");
    while (!feof(imf))
    {
      long photoID = 0;
      int elementsPerPoint = 0;
      int dimensionCount = 0;
      int pointCount = 0;
      int bytesPerElement = 0;
      fread(&photoID, 8, 1, imf); //photoID 8���ֽڣ�1��
      if (feof(imf)) break;
      fseek(imf, 16, SEEK_CUR);
      fread(&elementsPerPoint, 4, 1, imf); //elementsPerPoint 4���ֽڣ�1��
      fread(&dimensionCount, 4, 1, imf); // dimensionCount 4���ֽڣ�1��
      fread(&pointCount, 4, 1, imf); // dimensionCount 4���ֽڣ� 1��
      fread(&bytesPerElement, 4, 1, imf); // bytesPerElement 4���ֽڣ�1��
      fseek(imf, (elementsPerPoint * pointCount * bytesPerElement), SEEK_CUR); //�Ե�ǰλ�ã�������offset
      if (dimensionCount != nImageFeatures)
      {
        printf("Expected %d features but got %d\n", nImageFeatures, dimensionCount);
        exit(1);
      }
      fread(nodeFeatures[nodeIndex[photoID]], bytesPerElement, dimensionCount, imf);
    }
    fclose(imf);
    
    //���¶�ȡ����ļ��� ��r�ķ�ʽ��ȡ�� һ��ΪphotoID, ��������ȡ���photo��nImageFeatures
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

  int useGroupFeatures = config.GetBool("Data.useGroupFeatures"); // useGroupFeatures=true
  int useTagFeatures = config.GetBool("Data.useTagFeatures"); // useTagFeatures=false
  if (config.GetBool("Data.useSocialFeatures")) // useSocialFeatures=true
  {
    FILE* ef = fopen(edgeFile.c_str(), "r"); // edgeFile=data/trainingEdgeFeaturesPASCAL.txt
    int NE_;
    fscanf(ef, "%d %d", &NE_, &nEdgeFeatures); // �ļ���һ�� NE_=1716515 nEdgeFeatures=7
    
    NE = 0;
    int thisNode = 1;
    long prevID = -1;
    for (int e = 0; e < NE_; e ++) // ѭ����ȡedgeID1��edgeID2�Ķԣ�һ��NE_��
    {
      long nID1 = 0; long nID2 = 0;
      fscanf(ef, "%ld %ld", &nID1, &nID2); // ����ڶ��п�ʼ����ʽΪ 106998777 1361746217
      if (nID1 == prevID)
        thisNode ++;
      else
        thisNode = 1;
      prevID = nID1;
      
      double* feature = new double [nEdgeFeatures];
      // Ȼ���ٶ�ȡÿһ�к�����nEdgeFeatures��edgeFeature
      int total = 0;
      for (int j = 0; j < nEdgeFeatures; j ++)
      {
        float feat = 0;
        fscanf(ef, "%f", &feat);
        feature[j] = feat;
        if (j >= 2) total += feature[j] > 0 ? 1 : 0; // ��һ��ĺ��壿
      }
      if (nID1 == nID2) // ���edge����id��ͬ��Ϊ�쳣���˳�
      {
        printf("%ld %ld\n", nID1, nID2);
        printf("Got self loop\n");
        exit(1);
      }
      
      if (!useGroupFeatures) feature[0] = 0;
      if (!useTagFeatures) feature[1] = 0;
      // ����ÿ��id1(thisnode)�� ֻȡ���100��edge��Ϣ�� NEΪedgeIndex��ȫ���������
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
  // useSocialFeatures=true ûfirstOrderModel, ����δִ��
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

