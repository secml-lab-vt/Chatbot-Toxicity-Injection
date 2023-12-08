# Only Item to Modify
GPUs[0]=$3
SEED=71

YOUR_DIALOG_DATASET_NAME=$1 # fill in the dataset(folder) name defined in data_format.py
YOUR_DIALOG_MODEL_NAME=$2 # fill in the model(folder) name defined in data_format.py

# Return to main directory
cd ../

# Extract Keywords
bash extract_kw.sh $YOUR_DIALOG_DATASET_NAME $YOUR_DIALOG_MODEL_NAME

# Inference
bash ./script/inference_sh/Compute_GRADE_K2_N10_N10.sh $SEED ${GPUs[0]} $YOUR_DIALOG_DATASET_NAME $YOUR_DIALOG_MODEL_NAME
