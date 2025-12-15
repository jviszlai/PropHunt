#!/bin/bash

NUM_CORES="8"
CONTAINER_PATH="/usr/src/prop_hunt"
LOCAL_MOUNT="$PWD/scripts/data"
DOCKER_MOUNT="$CONTAINER_PATH/scripts/data"

# Hand Optimized Circuit Simulations

docker run --platform linux/amd64 -w ${CONTAINER_PATH}/scripts -v $LOCAL_MOUNT:$DOCKER_MOUNT \
prop_hunt python hand_opt_surface.py

# Introduction Data

docker run --platform linux/amd64 -w ${CONTAINER_PATH}/scripts -v $LOCAL_MOUNT:$DOCKER_MOUNT \
prop_hunt python intro_data.py

# PropHunt Evaluation

docker run --platform linux/amd64 -w ${CONTAINER_PATH}/scripts -v $LOCAL_MOUNT:$DOCKER_MOUNT \
prop_hunt python prophunt_experiment.py surface 3 100 $NUM_CORES

docker run --platform linux/amd64 -w ${CONTAINER_PATH}/scripts -v $LOCAL_MOUNT:$DOCKER_MOUNT \
prop_hunt python prophunt_experiment.py surface 5 100 $NUM_CORES

docker run --platform linux/amd64 -w ${CONTAINER_PATH}/scripts -v $LOCAL_MOUNT:$DOCKER_MOUNT \
prop_hunt python prophunt_experiment.py surface 7 250 $NUM_CORES

docker run --platform linux/amd64 -w ${CONTAINER_PATH}/scripts -v $LOCAL_MOUNT:$DOCKER_MOUNT \
prop_hunt python prophunt_experiment.py surface 9 300 $NUM_CORES

docker run --platform linux/amd64 -w ${CONTAINER_PATH}/scripts -v $LOCAL_MOUNT:$DOCKER_MOUNT \
prop_hunt python prophunt_experiment.py lp 3 150 $NUM_CORES

docker run --platform linux/amd64 -w ${CONTAINER_PATH}/scripts -v $LOCAL_MOUNT:$DOCKER_MOUNT \
prop_hunt python prophunt_experiment.py rqt 6 300 $NUM_CORES

docker run --platform linux/amd64 -w ${CONTAINER_PATH}/scripts -v $LOCAL_MOUNT:$DOCKER_MOUNT \
prop_hunt python prophunt_experiment.py rqt_di_156 4 200 $NUM_CORES

docker run --platform linux/amd64 -w ${CONTAINER_PATH}/scripts -v $LOCAL_MOUNT:$DOCKER_MOUNT \
prop_hunt python prophunt_experiment.py rqt_di_8020 4 200 $NUM_CORES

# Idle Sensitivity

docker run --platform linux/amd64 -w ${CONTAINER_PATH}/scripts -v $LOCAL_MOUNT:$DOCKER_MOUNT \
prop_hunt python idle_sensitivity.py

# Scaling Data

docker run --platform linux/amd64 -w ${CONTAINER_PATH}/scripts -v $LOCAL_MOUNT:$DOCKER_MOUNT \
prop_hunt python scaling_data.py