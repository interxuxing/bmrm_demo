/* Copyright (c) 2011, NICTA
 * All rights reserved.
 *
 * The contents of this file are subject to the Mozilla Public License Version
 * 1.1 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 * http://www.mozilla.org/MPL/
 *
 * Software distributed under the License is distributed on an "AS IS" basis,
 * WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
 * for the specific language governing rights and limitations under the
 * License.
 *
 */

#ifndef _L2N2_QLD_HPP_
#define _L2N2_QLD_HPP_

#include "sml.hpp"
#include "model.hpp"
#include "l2n2_bmrmdualinnersolver.hpp"
#include <iostream>
#include <cassert>


class CL2N2_qld : public CL2N2_BMRMDualInnerSolver 
{
public:
	CL2N2_qld(double lambda) : CL2N2_BMRMDualInnerSolver(lambda)
	{
	}

	virtual ~CL2N2_qld()
	{
	}

	/** Solve the QP */
	virtual void SolveQP();
};

#endif
