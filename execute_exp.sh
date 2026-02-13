# Submit jobs to SLURM.

ENV_NAME="tarte_test" # Change the environment name accordingly
JOB_NAME="strable_1"
TIME_HOUR=48
N_CPUS=32
MAX_PARALLEL_TASKS=40
PARTITION="normal-best" # parietal normal-best, normal gpu-best gpu
EXCLUDE="marg[033-035],margpu[002-003]"
PROBLEM_NAME=('all') # clear-corpus chocolate-bar-ratings
DTYPE_METHOD_NAME=("num-str") # "all" "num-str" "num-only" "str-only"
EMBED_METHOD_NAME=("llm-llama-3.1-8b" "llm-e5-small-v2")
ESTIM_METHOD_NAME=("xgb")
TUNE_INDICATOR=("tune") # "tune" or "default"
NS_VALUES=("3") # Fixed
FI_VALUES=("all") # all "0" "1" "2"
DEVICE="cpu" # cuda 'cpu'
CHECK_RES_FLAG="True"
OVERRIDE_CACHE="False"

# script_evaluate script_evaluate_test 
# script_extract_llm_embeddings

for i in ${PROBLEM_NAME[@]}; do
    conda run -n $ENV_NAME python -W ignore script_evaluate_normalize.py \
        -jn $JOB_NAME -t $TIME_HOUR --gpu -w $N_CPUS -mpt $MAX_PARALLEL_TASKS -p $PARTITION -ex $EXCLUDE \
        -dm ${DTYPE_METHOD_NAME[@]} -emm ${EMBED_METHOD_NAME[@]} -esm ${ESTIM_METHOD_NAME[@]} -ti ${TUNE_INDICATOR[@]}\
        -dn $i -ns ${NS_VALUES[@]} -fi ${FI_VALUES[@]} \
        -dv $DEVICE -cf $CHECK_RES_FLAG -oc $OVERRIDE_CACHE
done    