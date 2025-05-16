# Set name of experiment (file paths will update accordingly)
EXPERIMENT_NAME="qm9-homo-proponly-OOD-SELFIES-scratch-ood-test-eval"
# Set log directory
MY_LOG_DIR="/p/gpfs1/$USER/flask-experiments/EVAL/$EXPERIMENT_NAME/logs"
# Model save directory
MODEL_SAVE_DIR="/p/gpfs1/$USER/flask-experiments/EVAL/$EXPERIMENT_NAME/saved-model"
# Set output directory name based on current date
OUTPUT_DIRNAME="output/`date +"%Y-%m-%d"`"
# Create log directories if they don't exist
mkdir -p $MY_LOG_DIR
mkdir -p $MY_LOG_DIR/$OUTPUT_DIRNAME

###########################

MYBANK="flask"
MYTIME=720 # Job time in minutes (currently set to 12 hours)

echo $MY_LOG_DIR

# Eval
python ../../main.py \
  --model ../../../flask-experiments/qm9-homo-proponly-selfies-scratch-OOD_continue/logs/checkpoint-best-70000/ \
  --tokenizer ../../../flask-experiments/qm9-homo-proponly-selfies-scratch-OOD_continue/logs/checkpoint-best-70000/vocab.txt \
  --train-config ../../configs/train/qm9_homo_prop_only.json \
  --eval-only \
  --output-dir $MY_LOG_DIR \
  --eval-accumulation-steps 2 \
  --param-path ../../configs/eval/FLASK_qm9_homo_eval.json \
  --eval-file ../../OOD_data/qm9_homo_OOD/qm9_homo_OOD_ood_test.txt
