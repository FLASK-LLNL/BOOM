# Set name of experiment (file paths will update accordingly)
EXPERIMENT_NAME="qm9-zpve-proponly-selfies-scratch-OOD"
# Set log directory
MY_LOG_DIR="/p/gpfs1/$USER/flask-experiments/$EXPERIMENT_NAME/logs"
# Model save directory
MODEL_SAVE_DIR="/p/gpfs1/$USER/flask-experiments/$EXPERIMENT_NAME/saved-model"
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

bsub -J $EXPERIMENT_NAME -G $MYBANK -W $MYTIME -o $MY_LOG_DIR/$OUTPUT_DIRNAME/$EXPERIMENT_NAME-%J.out python ../../../main.py \
  --model ../../../pretrained_models/qed_IBM_scratch/ \
  --tokenizer ../../../pretrained_models/qed_IBM_scratch/vocab_qm9_zpve.txt \
  --train-data ../../../OOD_data/qm9_zpve_OOD/qm9_zpve_OOD_train.txt \
  --eval-data ../../../OOD_data/qm9_zpve_OOD/qm9_zpve_OOD_train.txt \
  --per-device-batch-size 8 \
  --eval-steps 1000 \
  --train-config ../../../configs/train/qm9_zpve_prop_only.json \
  --output-dir $MY_LOG_DIR \
  --learning-rate 1e-4 \
  --num-train-epochs 20 \
  --save-total-limit 2 \
  --save-steps 0 \
  --logging-steps 1000 \
  --eval-accumulation-steps 2
