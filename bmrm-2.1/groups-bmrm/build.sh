#!/bin/bash
set -e

#!/bin/bash

[ -h "genericloss.cpp" ] || ln -s ../loss/genericloss.cpp .
[ -h "genericloss.hpp" ] || ln -s ../loss/genericloss.hpp .
[ -h "genericdata.cpp" ] || ln -s ../data/genericdata.cpp .
[ -h "genericdata.hpp" ] || ln -s ../data/genericdata.hpp .

cd ../linear-bmrm && make all
cd ../groups-bmrm

[ -h "linear-bmrm-train" ] || ln -s ../linear-bmrm/linear-bmrm-train .
[ -h "linear-bmrm-predict" ] || ln -s ../linear-bmrm/linear-bmrm-predict .
