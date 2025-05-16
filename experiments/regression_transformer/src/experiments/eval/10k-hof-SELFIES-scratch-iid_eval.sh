# Set name of experiment (file paths will update accordingly)
EXPERIMENT_NAME="hof-proponly-10k-OOD-SELFIES-scratch-iid-test-eval"
# Set log directory
MY_LOG_DIR="/p/gpfs1/$USER/flask-experiments/EVAL/$EXPERIMENT_NAME/logs"
# Model save directory
MODEL_SAVE_DIR="/p/gpfs1/$USER/flask-experiments/EVAL/$EXPERIMENT_NAME/saved-model"
#MODEL_SAVE_DIR="/p/vast1/flaskdat/Models/regression-transformer/outputs/10k-den-SELFIES-proponly_OOD_finetune"
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
  --model ../../../flask-experiments/10k-hof-proponly-selfies-scratch-OOD/logs/checkpoint-best-78000/ \
  --tokenizer ../../pretrained_models/qed_IBM_pretrained/vocab.txt \
  --train-config ../../configs/train/hof_prop_only.json \
  --eval-only \
  --output-dir $MY_LOG_DIR \
  --eval-accumulation-steps 2 \
  --param-path ../../configs/eval/FLASK_hof_eval.json \
  --eval-file ../../OOD_data/10k_dft_hof_OOD/10k_dft_hof_OOD_iid_test.txt
