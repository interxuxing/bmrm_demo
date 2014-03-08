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

#ifndef _GENERICLOSS_HPP_
#define _GENERICLOSS_HPP_

#include "common.hpp"
#include "sml.hpp"
#include "data.hpp"
#include "genericdata.hpp"
#include "loss.hpp"
#include <string>
#include "timer.hpp"
#include "math.h"
#include <vector>

enum losstype
{
  ZEROONE,
  SCALED
};

/* Class for image annotation loss. */
class CGenericLoss : public CLoss
{
public:
  CGenericLoss(int dim, CModel* &model, CGenericData* &data);
  virtual ~CGenericLoss();

  // Interfaces
  void Usage() {}
  void ComputeLoss(double &loss)
  {
    throw CBMRMException("ERROR: not implemented!\n", "CGenericLoss::ComputeLoss()");
  }

  void ComputeLossAndGradient(double &loss, TheMatrix &grad);
  void Predict(CModel *model);
  void Evaluate(CModel *model);
  
  double LabelLoss(map<int,int>& y, map<int,int>& ybar, double lossPositive, double lossNegative, losstype lt);

  CGenericData* data;
protected:
};

#endif

