#!/bin/bash

src_config='./groups-bmrm/config_temp_airplane2.conf'
temp_dir='./groups-bmrm/mylog'
log_dir='/home/xuxing/Dropbox/Public/exp_bmrm'


for label in $(seq 26045 26060)
do
    echo 'now train classifier for '$label' '
    if [ "$label" -gt 26045 ]; then
	replace_id=$(expr $label - 1)	
	sed -i 's/'$replace_id'/'$label'/g' $src_config  	
    else
    	sed -i 's/Data.learnLabel/Data.learnLabel '$label'/g' $src_config
    fi

    # run the command
    ./linear-bmrm/linear-bmrm-train $src_config | tee $temp_dir/$label.log 
    # copy the log to dropbox directory
    cp $temp_dir/$label.log $log_dir/$label.log

done
    
echo 'finished training classifiers.'
