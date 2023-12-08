
!/usr/bin/env bash


function RUN_GRADE {
    YOUR_DIALOG_MODEL_NAME=$1
    RAW_TXT_FILE=$2
    YOUR_ROOT_DIR='/rdata/jiameng/GRADE/evaluation/eval_data/'
    YOUR_DIALOG_DATASET_NAME='mydata' # fill in the dataset(folder) name defined in data_format.py

    echo ${YOUR_DIALOG_MODEL_NAME}
    echo ${RAW_TXT_FILE}
    python ./data_format.py \
        --root_eval_dir $YOUR_ROOT_DIR \
        --dataset_name $YOUR_DIALOG_DATASET_NAME \
        --model_name $YOUR_DIALOG_MODEL_NAME \
        --raw_txt_file $RAW_TXT_FILE

    # Only Item to Modify
    GPUs[0]=1
    SEED=71

    # Return to main directory
    cd ../

    # Extract Keywords
    bash extract_kw.sh $YOUR_DIALOG_DATASET_NAME $YOUR_DIALOG_MODEL_NAME

    # Inference
    bash ./script/inference_sh/Compute_GRADE_K2_N10_N10.sh $SEED ${GPUs[0]} $YOUR_DIALOG_DATASET_NAME $YOUR_DIALOG_MODEL_NAME
}



#function run_trials {
#    MODEL_NAME=$1
#    CATEGORY=$2
#    EXP_ID=$3
#    PORTION_SIZE=$4
#    EVAL=$5
#
#
#
#    for TRIAL_IDX in 3
#    do
#        echo ${MODEL_NAME}
#        echo ${RAW_TXT_FILE}
#        echo "----------------"
#        GRADE ${MODEL_NAME} ${RAW_TXT_FILE}
#    done
#}

#script='/rdata/jiameng/GRADE/script/'
#pwd
#cd $script
#pwd


yourfilenames=`ls /rdata/crweeks/chatbot_security/chatbot-security-code/test_results/`
script='/rdata/jiameng/GRADE/script/'
for eachfile in $yourfilenames
do
   echo  $eachfile
   MODEL_NAME=$eachfile
   RAW_TXT_FILE='/rdata/crweeks/chatbot_security/chatbot-security-code/test_results/'$eachfile
   echo ${MODEL_NAME}
   echo ${RAW_TXT_FILE}
   echo "----------------"
   RUN_GRADE ${MODEL_NAME} ${RAW_TXT_FILE}
   cd $script
#   break
done