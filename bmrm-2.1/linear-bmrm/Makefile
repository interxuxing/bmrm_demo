# Set to 1 if COIN Clp is installed and you would like to use L1 norm regularizer
COIN_CLP_AVAILABLE=0
COIN_INC_DIR=../../Clp-1.11.1/include/coin
COIN_LIB_DIR=../../Clp-1.11.1/lib

# Set to 1 if BT code by Schramm and Zowe (1992) is avaialble and you would like to use it
BT_AVAILABLE=0

# Set to 1 if MPI (e.g. MPICH2) is installed and you would like to use parallelized bmrm
MPI_AVAILABLE=0


BOOST_DIR=../externalpackages/
UTILITIES_DIR=../utilities
DATA_DIR=../data
MODEL_DIR=../model
LOSS_DIR=../loss
SOLVER_DIR=../solver
INNER_SOLVER_DIR=../solver/bmrminnersolver

include ${LOSS_DIR}/loss.mk
include ${DATA_DIR}/data.mk
include ${SOLVER_DIR}/solver.mk
include ${UTILITIES_DIR}/utilities.mk
include ${MODEL_DIR}/model.mk

TRAINER = linear-bmrm-train

FACTORY_SRC = ${LOSS_DIR}/lossfactory.hpp ${DATA_DIR}/datafactory.hpp\
	      ${SOLVER_DIR}/solverfactory.hpp ${INNER_SOLVER_DIR}/bmrminnersolverfactory.hpp


INCLUDE_DIR = -I${DATA_DIR} -I${MODEL_DIR} -I${UTILITIES_DIR} -I${LOSS_DIR}\
	      -I${SOLVER_DIR} -I${INNER_SOLVER_DIR} -I${BOOST_DIR}

ALL_OBJECTS = ${UTILITIES_OBJ} ${DATA_OBJ} ${MODEL_OBJ} ${LOSS_OBJ} ${SOLVER_OBJ} ${INNER_SOLVER_OBJ}

CXX = g++

### for debugging ###
#CFLAGS = -Wall -ggdb -g ${INCLUDE_DIR}

### for release ###
CFLAGS = -Wall -O3 -DNDEBUG ${INCLUDE_DIR}

LDFLAGS = 

# use OpenMP for parallelization
CFLAGS += -fopenmp -DUSE_OPENMP
LDFLAGS  += -lgomp 

# use Intel's Math Kernel Library (MKL)
#CFLAGS += -DUSE_MKL
#LDFLAGS += -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread 
#CFLAGS += -DMKL -I $(INCLUDE)


LDFLAGS += -lgfortran

# use Apple's Accelerate framework
#CFLAGS += -DUSE_ACCELERATE
#LDFLAGS += -framework Accelerate


ifeq (${COIN_CLP_AVAILABLE},1)
	CFLAGS += ${COIN_CFLAGS} -DHAVE_L1N1_INNER_SOLVER
	LDFLAGS += ${COIN_LDFLAGS}
endif

ifeq (${BT_AVAILABLE},1)
	REGULARIZER_DIR=../regularizer
	include ${REGULARIZER_DIR}/regularizer.mk
	FACTORY_SRC += ${REGULARIZER_DIR}/regularizerfactory.hpp
	INCLUDE_DIR += -I${REGULARIZER_DIR}
	ALL_OBJECTS += ${REGULARIZER_OBJ}
	CFLAGS += -DHAVE_BT_SOLVER
	LDFLAGS += ${BT_LDFLAGS}
endif

ifeq (${MPI_AVAILABLE},1)
	# path to mpicxx
	CXX = /opt/mpich/gnu/bin/mpicxx
	CFLAGS += -DPARALLEL_BMRM
	LDFLAGS += -lmpich
	TRAINER = parallel-bmrm-train
endif


.PHONY: all

all: linear-bmrm-train linear-bmrm-predict

# executable
linear-bmrm-train: linear-bmrm-train.cpp ${ALL_OBJECTS} ${FACTORY_SRC}
	${CXX} ${CFLAGS} linear-bmrm-train.cpp -o ${TRAINER} ${ALL_OBJECTS} ${LDFLAGS}

linear-bmrm-predict: linear-bmrm-predict.cpp ${ALL_OBJECTS} ${FACTORY_SRC}
	${CXX} ${CFLAGS} linear-bmrm-predict.cpp -o linear-bmrm-predict ${ALL_OBJECTS} ${LDFLAGS}


clean:
	rm *.o *-bmrm-train linear-bmrm-predict;

