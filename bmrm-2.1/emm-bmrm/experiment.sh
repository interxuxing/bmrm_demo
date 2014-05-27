#!/bin/bash

for model in 1024;
do
  for lambda in 1e-1 1e-2 1e-3 1e-4;
  do
    rm -f model;
    cp configfiles/graphmatch.l2.empty.conf temp_train.conf;
    echo int Model.featDim $model >> temp_train.conf;
    echo string Data.inputPrefix desc_whole >> temp_train.conf;
    echo double BMRM.lambda $lambda >> temp_train.conf;
    # Run the main algorithm for 10 iterations
    for rep in `seq 1 10`;
    do
      ./gm-bmrm-train temp_train.conf | tee -a results/outputlog_$model_$lambda;
    done;
    # Run the algorithm on the validation set
    ./gm-bmrm-predict temp_train.conf | tee -a results/outputlog_$model_$lambda;
    # Run the algorithm on the test set
    echo string Data.inputPrefix desc_test >> temp_train.conf;
    ./gm-bmrm-predict temp_train.conf | tee -a results/outputlog_$model_$lambda;
    mv model results/model_$model_$lambda;
  done;
done;
rm temp_train.conf

for model in 4096;
do
  for lambda in 1e-5 1e-6 1e-7 1e-8;
  do
    rm -f model;
    cp configfiles/graphmatch.l2.empty.conf temp_train.conf;
    echo int Model.featDim $model >> temp_train.conf;
    echo string Data.inputPrefix desc_whole >> temp_train.conf;
    echo double BMRM.lambda $lambda >> temp_train.conf;
    # Run the main algorithm for 10 iterations
    for rep in `seq 1 10`;
    do
      ./gm-bmrm-train temp_train.conf | tee -a results/outputlog_$model_$lambda;
    done;
    # Run the algorithm on the validation set
    ./gm-bmrm-predict temp_train.conf | tee -a results/outputlog_$model_$lambda;
    # Run the algorithm on the test set
    echo string Data.inputPrefix desc_test >> temp_train.conf;
    ./gm-bmrm-predict temp_train.conf | tee -a results/outputlog_$model_$lambda;
    mv model results/model_$model_$lambda;
  done;
done;
rm temp_train.conf
