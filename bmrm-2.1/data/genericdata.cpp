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
  if (config.GetBool("Prediction.outputFvalAndLabels"))  // null, 这里是用在test集合上做验证的, 原始的conf文件给定的是train+test
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
  if (config.GetBool("Data.useImageFeatures")) // false, demo代码中没有用到imagefeature
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
  // idFile =  data/trainingGroupIDsPASCAL.txt, 这个文件可以不用改， 因为它包含了几个dataset的group信息
  FILE* idf = fopen(idFile.c_str(), "r");
  int currentID;
  vector<int> groupsToUse;
  vector<int> tagsToUse;
  set<int> groupsToUseSet;
  set<int> tagsToUseSet;
  int NGroups_;
  int NTags_;
  while (fscanf(idf, "%d", &currentID) == 1) // 每次循环取一个currentID, 每一行第一个值，比如第一行的 26041
  {
    char* name = new char [1000];
    fscanf(idf, "%s %d %d", name, &NGroups_, &NTags_);
    // 比如 PASCAL:aeroplane 200 0 -> name=PASCAL:aeroplane, NGroups_=200 NTags_=0
    // 每一行后续成对的数据x:f 按照先NGroups_，后NTags_的顺序存放
    for (int i = 0; i < NGroups_; i ++)
    {
      int x;
      float f;
      fscanf(idf, "%d:%f", &x, &f);
      if (currentID == learnLabel) // learnLabel=6043为aeroplane, 变更这个值， 可以选择不同的数据
        if (f > 2) // f大于2才存
        {
          groupsToUse.push_back(x); // vector用push_back
          groupsToUseSet.insert(x); // set用insert
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
    if (currentID == learnLabel) break; // 只选learnLabel这一个
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
  // textIdFile = data/trainingWordIDsPASCAL.txt, 这个文件不用改， 它包含了几个dataset的word信息
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
      fscanf(tidf, "%d:%f", &x, &f); // 这个x:f格式和之前的idf那个文件中groups/tags类似
      if (currentID == learnLabel)
      {
        if (f > 2 and not (config.GetBool("Data.baseline"))) //f>2是指frequency>2 ?
        {
          wordsToUse.push_back(x);
          wordsToUseMap[x] = wordId ++; // x是word的id， wordId++是把出现次数累加，这里的map是<x, wordId>
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
  // nodeFile = data/trainingIndicatorsPASCAL.txt, 这个文件对应于nodeFeaturesXXX.txt文件
  FILE* nf = fopen(nodeFile.c_str(), "r");
  // nodeFile这个文件， 第一行存放总体信息， groups, tags, labels的数目
  fscanf(nf, "%d %d %d", &NGroups, &NTags, &NLabels); // NGroups=6041 NTags=20000 NLabels=215
  // 从第二行开始， 先存完groups, 然后tags， 最后labels
  // groups的每一行的格式为: id, name, 比如0 20759249@N00
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
  // 前面三个参数ngroups, ntags, nlabels读取完以后，紧接着一行(第6041+20000+215+1行)是8151，为nphotos的个数
  fscanf(nf, "%d", &NPhotos);
  
  NGroups_ = groupsToUse.size();
  NTags_ = tagsToUse.size();
  NWords_ = wordsToUse.size();

  if (!config.GetBool("Data.useGroupFeatures")) // useGroupFeatures=true
  {
    NGroups_= 0; //未执行
  }
  if (!config.GetBool("Data.useTagFeatures")) // useTagFeatures=false
  {
    NTags_ = 0; // 执行了
  }
  
  if (config.GetBool("Data.baseline")) // baseline=false
  {
    NGroups_ = 0; // 未执行
    NWords_ = 0;
  }

  if (config.GetBool("Data.useNodeFeatures")) // useNodeFeatures=true
    nNodeFeatures = nImageFeatures + NGroups_ + NTags_ + NWords_ + NOthers; // 所有weight参数的总数
  else
  {
    nNodeFeatures = nImageFeatures;
    NGroups_ = 0;
    NTags_ = 0;
    NWords_ = 0;
  }

  // 这里开始读取nphotos代表的features
  // Scale features
  lossPositive = 0;
  lossNegative = 0;
  for (int i = 0; i < NPhotos; i ++)
  {
    long photoId = 0;
    char* userId = new char [50];
    char* indicator = new char [NGroups + NTags + NLabels + 2];
    // 这里每一行的存放格式为 111604571 52734910@N00 ....带很长的indicators, 注意每个photo都是有NGroups + NTags + NLabels + 2维度的indicator的
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
          feature[nImageFeatures + i] = 1.0/NGroups_; //归一化
        else feature[nImageFeatures + i] = 0;
      }
      for (int i = 0; i < NTags_; i ++)
      {
        if (indicator[tagsToUse[i]] == '1')
          feature[nImageFeatures + NGroups_ + i] = 1.0/NTags_; //归一化
        else feature[nImageFeatures + NGroups_ + i] = 0;
      }
    }

    nodeFeatures[i] = feature; // 每一个photo的nodeFeatures的维度是[nImageFeatures+NGroups_+NTags_+NWords_+NOthers]
    nodeLabels[i] = NEGATIVE; // nodeLabel大小是[NPhotos], 初始都置为-1， 如果learnLabel指定， 则置为+1
    if (indicator[learnLabel] == '1') nodeLabels[i] = POSITIVE; //indicator大小是[NGroups+NTags+NLabels+2], 从indicator那个文件中读取的
    if (indicator[learnLabel] == '0') nodeLabels[i] = NEGATIVE;
    if (nodeLabels[i] == POSITIVE) lossPositive ++; //统计positive的个数
    if (nodeLabels[i] == NEGATIVE) lossNegative ++; //统计negative的个数

    delete [] userId;
    delete [] indicator;
  }
  printf("For learnLabel: %d, num of positive: %f, negative: %f \n", learnLabel, lossPositive, lossNegative);
  fclose(nf);

  if (config.GetBool("Data.useNodeFeatures")) // useNodeFeatures=true
  {
	  // 读取data/trainingTextPASCAL.txt这个文件
    FILE* tf = fopen(textFile.c_str(), "r"); // textFile=data/trainingTextPASCAL.txt
    int NWords;
    // 文件第一行是统计words的综述NWords=14227
    fscanf(tf, "%d", &NWords);
    // 之后逐行读取 wordID:word的格式, 比如6 shot, 注意这里读取的wordID:word根本没有使用！
    for (int i = 0; i < NWords; i ++)
    {
      int wordID;
      char* word = new char [1000];
      fscanf(tf, "%d %s", &wordID, word);
      delete [] word;
    }
    // 上面word读取完以后， 接下来是photos的存储， 开始一行(14229行)格式为8151111604571 52734910@N00 3 ...
    // 这里对应了每个photo和userid的间的关系， 每个wordid出现的次数
    for (int i = 0; i < NPhotos; i ++)
    {
      long photoId = 0;
      char* userId = new char [1000];
      int nf;
      // 8151111604571 52734910@N00 3表示 photoID, userID, nf
      if (not fscanf(tf, "%ld %s %d", &photoId, userId, &nf))
      {
        printf("Expected ID and int\n");
        exit(1);
      }
      // feature这个指针定位到photo的开始处
      double* feature = nodeFeatures[nodeIndex[photoId]] + nImageFeatures + NGroups_ + NTags_;
      for (int i = 0; i < NWords_; i ++)
      {
        feature[i] = 0;
      }
      // 然后读取nf个photo feature， 比如第14229行， nf=3 后续674:1 5908:1 63:1， 表示某个word出现的次数
      for (int i = 0; i < nf; i ++)
      {
        int hkey, hvalue;
        fscanf(tf, "%d:%d", &hkey, &hvalue);
        if (wordsToUseMap.find(hkey) != wordsToUseMap.end())
          feature[wordsToUseMap[hkey]] = hvalue * 1.0/NWords_; //归一化
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

  // trainingEvidence=false 没有outputFvalAndLabels, 以下不执行
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

  // 此段是读取imagefeature文件的， 但是此demo中useImageFeatures=false, 以下程序段未执行
  if (config.GetBool("Data.useImageFeatures"))
  {
	  //从此处分析看， imageFeature应该是存放在单一一个文件中， 每一行为一个photo的feature
    imf = fopen(imageFeatures.c_str(), "rb");
    while (!feof(imf))
    {
      long photoID = 0;
      int elementsPerPoint = 0;
      int dimensionCount = 0;
      int pointCount = 0;
      int bytesPerElement = 0;
      fread(&photoID, 8, 1, imf); //photoID 8个字节，1个
      if (feof(imf)) break;
      fseek(imf, 16, SEEK_CUR);
      fread(&elementsPerPoint, 4, 1, imf); //elementsPerPoint 4个字节，1个
      fread(&dimensionCount, 4, 1, imf); // dimensionCount 4个字节，1个
      fread(&pointCount, 4, 1, imf); // dimensionCount 4个字节， 1个
      fread(&bytesPerElement, 4, 1, imf); // bytesPerElement 4个字节，1个
      fseek(imf, (elementsPerPoint * pointCount * bytesPerElement), SEEK_CUR); //以当前位置，往后增offset
      if (dimensionCount != nImageFeatures)
      {
        printf("Expected %d features but got %d\n", nImageFeatures, dimensionCount);
        exit(1);
      }
      fread(nodeFeatures[nodeIndex[photoID]], bytesPerElement, dimensionCount, imf);
    }
    fclose(imf);
    
    //重新读取这个文件， 以r的方式读取， 一行为photoID, 接下来读取这个photo的nImageFeatures
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
    fscanf(ef, "%d %d", &NE_, &nEdgeFeatures); // 文件第一行 NE_=1716515 nEdgeFeatures=7
    
    NE = 0;
    int thisNode = 1;
    long prevID = -1;
    for (int e = 0; e < NE_; e ++) // 循环读取edgeID1和edgeID2的对，一共NE_行
    {
      long nID1 = 0; long nID2 = 0;
      fscanf(ef, "%ld %ld", &nID1, &nID2); // 比如第二行开始，格式为 106998777 1361746217
      if (nID1 == prevID)
        thisNode ++;
      else
        thisNode = 1;
      prevID = nID1;
      
      double* feature = new double [nEdgeFeatures];
      // 然后再读取每一行后续的nEdgeFeatures个edgeFeature
      int total = 0;
      for (int j = 0; j < nEdgeFeatures; j ++)
      {
        float feat = 0;
        fscanf(ef, "%f", &feat);
        feature[j] = feat;
        if (j >= 2) total += feature[j] > 0 ? 1 : 0; // 这一句的含义？
      }
      if (nID1 == nID2) // 如果edge两个id相同，为异常，退出
      {
        printf("%ld %ld\n", nID1, nID2);
        printf("Got self loop\n");
        exit(1);
      }
      
      if (!useGroupFeatures) feature[0] = 0;
      if (!useTagFeatures) feature[1] = 0;
      // 对于每个id1(thisnode)， 只取最多100个edge信息， NE为edgeIndex的全局索引编号
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
  // useSocialFeatures=true 没firstOrderModel, 以下未执行
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

