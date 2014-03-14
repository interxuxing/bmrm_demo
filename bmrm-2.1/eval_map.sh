#!/bin/bash

src_config='./groups-bmrm/config_temp_airplane2.conf'
temp_dir='./groups-bmrm/mylog'

num_label=0
map=0

for label in $(seq 26041 26045)
do
    # find the AP value in each log file
    num_label=$(expr $num_label + 1)	
    ap=$(eval cat $temp_dir/$label.log | grep 'AvP' | sed 's/^AvP = //g')
    echo 'current ap is '$ap''
    map=$(echo "$map + $ap" | bc -l)
    echo 'map is '$map''
done

# now calculate the map value for all labels
map=$(echo "$map / $num_label" | bc -l)
echo 'map for all '$num_label' labels is '$map''
