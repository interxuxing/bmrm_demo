/* Copyright (c) 2010, (removed for blind review)
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

#ifndef _GENERICDATA_HPP_
#define _GENERICDATA_HPP_

#include <vector>
#include "common.hpp"
#include "sml.hpp"
#include "data.hpp"
#include "configuration.hpp"

using namespace std;

#define Cost float

Cost inner_product(double* x, double* y, int D);

enum labelType
{
  EVIDENCE_NEGATIVE = -2, // A negative evidence node
  NEGATIVE = -1, // A latent variable whose correct label is negative
  UNKNOWN = 0, // A variable whose correct labels is unknown
  POSITIVE = 1, // A latent variable whose correct label is postive
  EVIDENCE_POSITIVE = 2 // A positive evidence node
};

class CGenericData: public CData
{
public:
  CGenericData();
  virtual ~CGenericData();
  
  int NPhotos; // Total number of photos
  int NE; // Total number of edges

  int NGroups; // Total number of groups
  int NTags; // Total number of tags
  int NLabels; // Total number of labels
  map<int, string> labelNames; // Human-readable label names
  int learnLabel; // Which label is the one for which we are learning a model
  double lossPositive; // loss incurred on positively labeled images
  double lossNegative; // loss incurred on negatively labeled images

  map<long, int> nodeIndex; // Flickr id to integer id
  map<int, long> indexNode; // int id to Flickr id
  map<pair<int,int>, int> edgeIndex; // edge id
  map<int, pair<int,int> > indexEdge; // reverse edge id

  map<int, int> nodeLabels; // training labels of each node

  // nodeFeatures + edgeFeatures为最终的输入feature
  map<int, double*> nodeFeatures; // 节点本身的feature
  map<int, double*> edgeFeatures; // 节点间的pairwise feature
  
  map<int, double>* firstOrderResponses;

  int num_positive; // Number of positive constraints needed to enforce submodularity

  virtual bool         bias(void)       const { return false; }
  virtual unsigned int slice_size(void) const { return N; }
  virtual unsigned int size(void)       const { return N; }
  virtual unsigned int dim(void)        const 
  { 
    return nNodeFeatures + nEdgeFeatures;
  }
  int nNodeFeatures; // Node feature dimensionality
  int nEdgeFeatures; // Edge feature dimensionality
  int N;
private:
  CGenericData(const CGenericData&);
};

#endif

