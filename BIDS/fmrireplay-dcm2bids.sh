PATH_BASE="${HOME}"
echo ${PATH_BASE}
# path to the project root directory
PATH_ROOT="${PATH_BASE}/fMRIreplay_hq/fmrireplay-bids"
# define the name of the project:
PROJECT_NAME="BIDS-nofieldmap"
# define the path to the project folder:
PATH_PROJECT="${PATH_ROOT}/${PROJECT_NAME}"
# define the path to the input directory:
PATH_INPUT="${PATH_ROOT}/BIDS/sourcedata/fmri2"
# define the path to the output directory
PATH_OUTPUT="${PATH_PROJECT}"
PATH_CONFIG="${PATH_PROJECT}/code/BIDS"
# path to the text file with all subject ids:
PATH_SUB_LIST="${PATH_ROOT}/BIDS/code/fmrireplay-new-participant-list.txt"
# maximum number of cpus per process:
N_CPUS=8
# memory demand in *GB*
MEM_GB=20
# read subject ids from the list of the text file
SUB_LIST=$(cat ${PATH_SUB_LIST} | tr '\n' ' ')
# ==============================================================================
# RUN HEUDICONV:
# ==============================================================================
# initalize a subject counter:
SUB_COUNT=0
# loop over all subjects:
for SUB in ${SUB_LIST}; do
	# update the subject counter:
	let SUB_COUNT=SUB_COUNT+1
	# get the subject number with zero padding:
	SUB_PAD=$(printf "%02d\n" $SUB_COUNT)
	echo "#!/bin/bash" > job
	# define the heudiconv command:
	echo "dcm2bids -d ${PATH_INPUT}/sub-${SUB} \
	-p ${SUB} \
	-c $PATH_CONFIG/config_sub-${SUB}.json \
	-o $PATH_OUTPUT \
	--forceDcm2niix" >> job

	sbatch job
	# rm -f job

done
