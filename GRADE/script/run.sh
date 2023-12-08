YOUR_ROOT_DIR='/rdata/jiameng/GRADE/evaluation/eval_data/'
YOUR_DIALOG_DATASET_NAME='mydata' # fill in the dataset(folder) name defined in data_format.py
YOUR_DIALOG_MODEL_NAME=$1 # fill in the model(folder) name defined in data_format.py
RAW_TXT_FILE=$2

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