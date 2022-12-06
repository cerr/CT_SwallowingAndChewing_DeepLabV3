#!/bin/bash

REPOREMOTE="https://github.com/aditiiyer/CT_SwallowingAndChewing_DeepLabV3.git"

#get repo commit hash to stamp output SIF filename
IDSTAMP=`git log | grep commit | head -1 | awk '{ print $2 }' | cut -c1-5`
#create txt file with hash information
echo "${IDSTAMP}" > model_wrapper/hash_id.txt

export RECIPE_FILE=sing_container_recipe.def
export SIFPATH=CT_SwallowingAndChewing_DeepLabV3_Container.sif
export SINGULARITY_TMPDIR=/singularity

dzdo singularity build --tmpdir=${SINGULARITY_TMPDIR} ${SIFPATH} ${RECIPE_FILE}

chmod 775 ${SIFPATH}

