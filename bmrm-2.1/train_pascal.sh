#!/bin/bash

src_config='./groups-bmrm/config_temp_airplane2.conf'
temp_dir='./groups-bmrm/mylog'
log_dir='/home/xuxing/Dropbox/Public/exp_bmrm'

prefix_labeloutput='groups-bmrm\/labels\/labels_'
prefix_predoutput='groups-bmrm\/preds\/pred_'
prefix_modelfile='groups-bmrm\/models\/mod_'

for label in $(seq 26041 26060)
do
    echo 'now train classifier for '$label' '
    if [ "$label" -gt 26041 ]; then
	replace_id=$(expr $label - 1)	
	sed -i 's/'$replace_id'/'$label'/g' $src_config  	
    else
    	sed -i 's/Data.learnLabel/Data.learnLabel '$label'/g' $src_config
	sed -i 's/Data.labelOutput/Data.labelOutput '$prefix_labeloutput''$label'.txt/g' $src_config
	sed -i 's/Data.predOutput/Data.predOutput '$prefix_predoutput''$label'.txt/g' $src_config
	sed -i 's/Model.modelFile/Model.modelFile '$prefix_modelfile''$label'.txt/g' $src_config
    fi

    # run the command
    ./linear-bmrm/linear-bmrm-train $src_config | tee $temp_dir/$label.log 
    # copy the log to dropbox directory
    cp $temp_dir/$label.log $log_dir/$label.log

done
    
echo 'finished training classifiers.'
