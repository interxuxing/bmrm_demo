#ifndef _GRAPHMATCHLOSS_HPP_
#define _GRAPHMATCHLOSS_HPP_

#include "common.hpp"
#include "sml.hpp"
#include "data.hpp"
#include "graphdata.hpp"
#include "loss.hpp"
#include <string>
#include "timer.hpp"

#include <boost/numeric/ublas/matrix.hpp>

void assignment(float* imFeatures,
                Scalar* w,
                float** indicator,
                int* classes,
                int maxviolator,
                int label,
                int** lossMatrix,
                float* preProduct,
                unsigned int* prePa);

/** Class for graphmatch classification loss.
 */
class CGraphMatchLoss : public CLoss
{
  public:
    CGraphMatchLoss(CModel* &model, CGraphData* &data);
    virtual ~CGraphMatchLoss();

    // Interfaces
    void Usage();
    void ComputeLoss(Scalar& loss)
    {
      throw CBMRMException("ERROR: not implemented!\n", "CGraphMatchLoss::ComputeLoss()");
    }
    void ComputeLossAndGradient(Scalar& loss, TheMatrix& grad);
    void Predict(CModel *model);
    void Evaluate(CModel *model);

    Scalar LabelLoss(int label, int* classes, int N);

    void LoadModel(std::string modelFilename="");
    void SaveModel(std::string modelFilename="");

  protected:
    void Phi(float* imFeatures, int* classes, int N, int* latent, int L, float* res);

    CGraphData* _data;

    int quad;
};

#endif
