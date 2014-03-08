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

#include "genericdata.hpp"
#include "genericloss.hpp"
#include "genericdata.hpp"
#include "configuration.hpp"
#include "math.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "graph.h"
#include "unistd.h"

#include <omp.h>

// Use the balanced error rate
#define LOSS SCALED

double loss1(int l, int lhat, double lossPositive, double lossNegative, losstype lt)
{
  if (l == lhat) return 0;
  if (l == UNKNOWN or l == EVIDENCE_POSITIVE or l == EVIDENCE_NEGATIVE) return 0;
  if (lt == ZEROONE) return 1.0 / (lossPositive + lossNegative);
  
  if (lt == SCALED and l == POSITIVE) return (1/2.0) / lossPositive;
  if (lt == SCALED and l == NEGATIVE) return (1/2.0) / lossNegative;
  printf("Souldn't be here\n");
  exit(1);
  return -1;
}

void setNodeCost(Graph<Cost,Cost,Cost>* g, int i, Cost E0, Cost E1)
{
  if (E0 > E1)
    g->add_tweights(i, E1 - E0, 0);
  else
    g->add_tweights(i, 0, E0 - E1);
}

// Convert edge costs to costs for the graph cuts objective
void setEdgeCost(Graph<Cost,Cost,Cost>* g, int i, int j, Cost E00, Cost E01, Cost E10, Cost E11)
{
  Cost A = E00;
  Cost B = E01;
  Cost C = E10;
  Cost D = E11;

  g->add_edge(i, j, B + C - A - D, 0);

  if (C > A)
    g->add_tweights(i, C - A, 0);
  else
    g->add_tweights(i, 0, A - C);
  
  if (C > D)
    g->add_tweights(j, 0, C - D);
  else
    g->add_tweights(j, D - C, 0);
}

// Column generation and prediction
void minimize(map<int, double*>& nodeFeatures,
              map<int,int>* nodeLabels,
              map<int, double*>& edgeFeatures,
              double* nodeTheta,
              double* edgeTheta,
              map<int,int>& res,
              int DN,
              int DE,
              double lossPositive,
              double lossNegative,
              map<int, pair<int,int> >& indexEdge,
              pair<int,double>* confidences,
              int colgen,
              map<int, double>* firstOrderResponses
             )
{
  int N = nodeFeatures.size();
  int E = edgeFeatures.size();
  
  typedef Graph<Cost,Cost,Cost> GraphType;
  GraphType* g = new GraphType(N, E);
  
  for (int i = 0; i < N; i ++)
    g->add_node();
  
  for (map<int, double*>::iterator it = nodeFeatures.begin(); it != nodeFeatures.end(); it ++)
  {
    Cost c = inner_product(it->second, nodeTheta, DN);
    if (confidences)
    {
      confidences[it->first].first = NEGATIVE;
      if (firstOrderResponses)
        confidences[it->first].second = firstOrderResponses->at(it->first);
      else
        confidences[it->first].second = c;
    }

    if (nodeLabels->at(it->first) == EVIDENCE_NEGATIVE) c = -1000;
    if (nodeLabels->at(it->first) == EVIDENCE_POSITIVE) c = 1000;

    double l0 = 0;
    double l1 = 0;
    if (colgen)
    {
      l0 = loss1(nodeLabels->at(it->first), NEGATIVE, lossPositive, lossNegative, LOSS);
      l1 = loss1(nodeLabels->at(it->first), POSITIVE, lossPositive, lossNegative, LOSS);
    }

    setNodeCost(g, it->first, c - l0, -c - l1);

    if (DE == 0)
    {
      if (c - l0 <= -c - l1)
        res[it->first] = NEGATIVE;
      else
        res[it->first] = POSITIVE;
      if (nodeLabels->at(it->first) == EVIDENCE_NEGATIVE)
        res[it->first] = EVIDENCE_NEGATIVE;
      if (nodeLabels->at(it->first) == EVIDENCE_POSITIVE)
        res[it->first] = EVIDENCE_POSITIVE;
    }
  }

  if (not DE)
  {
    delete g;
    return;
  }

  Cost* innerProducts00 = new Cost [E];
  
  #pragma omp parallel for
  for (int i = 0; i < E; i ++)
  {
    innerProducts00[i] = inner_product(edgeFeatures[i], edgeTheta, DE);
  }

  for (map<int, double*>::iterator it = edgeFeatures.begin(); it != edgeFeatures.end(); it ++)
  {    
    Cost c00 = innerProducts00[it->first];
    pair<int,int> edge = indexEdge[it->first]; 


    setEdgeCost(g, edge.first, edge.second, -c00, 0, 0, -c00);
  }
  
  delete [] innerProducts00;

  g->maxflow();
  
  for (int i = 0; i < N; i ++)
  {
    if (g->what_segment(i) == GraphType::SOURCE)
      res[i] = NEGATIVE;
    else
      res[i] = POSITIVE;
    if (nodeLabels->at(i) == EVIDENCE_POSITIVE)
    {
      if (res[i] != POSITIVE)
        printf("Label for node %d should be positive\n", i);
      res[i] = EVIDENCE_POSITIVE;
    }
    if (nodeLabels->at(i) == EVIDENCE_NEGATIVE)
    {
      if (res[i] != NEGATIVE)
        printf("Label for node %d should be negative\n", i);
      res[i] = EVIDENCE_NEGATIVE;
    }
    
    if (confidences)
      confidences[i].first = res[i];
  }

  delete g;
}

void Phi(map<int, double*>& nodeFeatures, map<int,int>* nodeLabels, map<int, double*>& edgeFeatures, int DN, int DE, double* nodePhi, double* edgePhi, map<int, pair<int,int> > indexEdge)
{
  for (int i = 0; i < DN; i ++)
    nodePhi[i] = 0;
  for (int i = 0; i < DE; i ++)
    edgePhi[i] = 0;

  for (map<int, double*>::iterator it = nodeFeatures.begin(); it != nodeFeatures.end(); it ++)
  {
    if (nodeLabels->at(it->first) == POSITIVE)
      for (int i = 0; i < DN; i ++)
        nodePhi[i] += it->second[i];
    else if (nodeLabels->at(it->first) == NEGATIVE)
      for (int i = 0; i < DN; i ++)
        nodePhi[i] -= it->second[i];
  }

  for (map<int, double*>::iterator it = edgeFeatures.begin(); it != edgeFeatures.end(); it ++)
  {
    pair<int,int> edge = indexEdge[it->first];
    int l0 = nodeLabels->at(edge.first);
    int l1 = nodeLabels->at(edge.second);
    
    if (fabs(l0) > 1.5 and fabs(l1) > 1.5) // both evidence nodes
      continue;

    if (l0 < 0 and l1 < 0) // negative agreement
      for (int i = 0; i < DE; i ++)
        edgePhi[i] += it->second[i];
    if (l0 > 0 and l1 > 0) // positive agreement
      for (int i = 0; i < DE; i ++)
        edgePhi[i] += it->second[i];
  }
}

void CGenericLoss::ComputeLossAndGradient(double& loss, TheMatrix& grad)
{
  loss = 0;
  grad.Zero();
  TheMatrix &w = _model->GetW();
  double* dat = w.Data();
  double* raw_g = grad.Data();

  {
    double* resy;
    double* resybar;

    map<int,int> ybar;

    resy = new double [data->dim()];
    resybar = new double [data->dim()];

    minimize(data->nodeFeatures, &(data->nodeLabels), data->edgeFeatures, dat, dat + data->nNodeFeatures, ybar, data->nNodeFeatures, data->nEdgeFeatures, data->lossPositive, data->lossNegative, data->indexEdge, NULL, 1, data->firstOrderResponses);

    Phi(data->nodeFeatures, &(data->nodeLabels), data->edgeFeatures, data->nNodeFeatures, data->nEdgeFeatures, resy,    resy    + data->nNodeFeatures, data->indexEdge);
    Phi(data->nodeFeatures, &ybar,               data->edgeFeatures, data->nNodeFeatures, data->nEdgeFeatures, resybar, resybar + data->nNodeFeatures, data->indexEdge);
    
    loss += LabelLoss(data->nodeLabels, ybar, data->lossPositive, data->lossNegative, LOSS);

    for (int j = 0; j < (int) data->dim(); j ++)
    {
      loss += dat[j]*(resybar[j]-resy[j]);
      raw_g[j] += (1.0/data->N)*(resybar[j]-resy[j]);
    }

    delete [] resy;
    delete [] resybar;
  }

  loss = loss/data->N;
}

CGenericLoss::CGenericLoss(int dim, CModel* &model_, CGenericData* &data_) : CLoss(model_, 1, dim, data_->bias()), data(data_)
{
  srand(0);
  Configuration &config = Configuration::GetInstance();

  verbosity = config.GetInt("Loss.verbosity");
}

void CGenericLoss::Predict(CModel *model)
{
  printf("In %s!\n", __FUNCTION__);
  Evaluate(model);
}

CGenericLoss::~CGenericLoss(void)
{
}

bool sortconf(pair<pair<int,double>,int> i, pair<pair<int,double>,int> j)
{
  if (i.first.first != j.first.first) return i.first.first > j.first.first;
  if (i.first.second > j.first.second) return true;
  return false;
}


double AvP(pair<int,double>* confidences_, map<int,int> gt_)
{
  map<int,int> gt;
  vector<pair<int,double> > confidences;
  int x = 0;
  for (int i = 0; i < (int) gt_.size(); i ++)
  {
    if (gt_[i] == POSITIVE or gt_[i] == NEGATIVE)
    {
      gt[x++] = gt_[i];
      confidences.push_back(confidences_[i]);
    }
  }
  
  int N = gt.size();
  vector<pair<pair<int,double>,int> > confsort;
  for (int i = 0; i < N; i ++)
    confsort.push_back(pair<pair<int,double>,int>(confidences[i],i));
  sort(confsort.begin(), confsort.end(), sortconf);
  double* rec = new double [N];
  double* prec = new double [N];
  double cumsum = 0;
  double nPositive = 0;
  for (int i = 0; i < N; i ++)
    if (gt[i] == POSITIVE) nPositive ++;
  for (int i = 0; i < N; i ++)
  {
    cumsum += gt[confsort[i].second] == POSITIVE ? 1 : 0;
    rec[i] = cumsum / nPositive;
    prec[i] = cumsum / (i + 1);
  }
  
  double ap = 0;
  for (int t = 0; t <= 10; t ++)
  {
    double p = 0;
    for (int i = 0; i < N; i ++)
      if (rec[i] >= t/10.0 and prec[i] >= p)
        p = prec[i];
    ap += p/11;
  }
  return ap;
}

void CGenericLoss::Evaluate(CModel *model)
{
  TheMatrix &w = _model->GetW(); 
  double* dat = w.Data();

  Configuration &config = Configuration::GetInstance();
  string outFile = config.GetString("Data.labelOutput");
  FILE* f = fopen(outFile.c_str(), "w");

  {
    map<int,int> ybar;
    pair<int,double>* confidences = new pair<int,double> [data->nodeFeatures.size()];
    minimize(data->nodeFeatures, &(data->nodeLabels), data->edgeFeatures, dat, dat + data->nNodeFeatures, ybar, data->nNodeFeatures, data->nEdgeFeatures, data->lossPositive, data->lossNegative, data->indexEdge, confidences, 0, data->firstOrderResponses);

    printf("0/1 loss    = %f\n", LabelLoss(data->nodeLabels, ybar, data->lossPositive, data->lossNegative, ZEROONE));
    printf("scaled loss = %f\n", LabelLoss(data->nodeLabels, ybar, data->lossPositive, data->lossNegative, SCALED));
    printf("AvP = %f\n", AvP(confidences, data->nodeLabels));
    printf("loss = %f\n", LabelLoss(data->nodeLabels, ybar, data->lossPositive, data->lossNegative, LOSS));
    delete [] confidences;

    for (map<int,int>::iterator it = ybar.begin(); it != ybar.end(); it ++)
      if (it->second == 1)
        fprintf(f, "%ld\n", data->indexNode[it->first]);
  }
  
  fclose(f);
}

double CGenericLoss::LabelLoss(map<int,int>& y, map<int,int>& ybar, double lossPositive, double lossNegative, losstype lt)
{
  double l = 0;
  for (map<int,int>::iterator it = y.begin(); it != y.end(); it ++)
    l += loss1(it->second, ybar[it->first], lossPositive, lossNegative, lt);
  return l;
}

