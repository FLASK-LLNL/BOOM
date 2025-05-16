# Set name of experiment (file paths will update accordingly)
EXPERIMENT_NAME="qm9-r2-proponly-OOD-SELFIES-scratch-ood-test-eval"
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
#  --model /usr/workspace/flask/experiments/RT-2.0/models/den-proponly-10k/checkpoint-best-14600 \
#  --model /usr/workspace/flask/RT-models/qed_IBM_pretrained \
#  --tokenizer /usr/workspace/flask/experiments/RT-2.0/models/den-proponly-10k/checkpoint-best-14600 \
#python scripts/eval_language_modeling.py \
#echo bsub -J EVAL-$EXPERIMENT_NAME -G $MYBANK -W $MYTIME -o $MY_LOG_DIR/$OUTPUT_DIRNAME/$EXPERIMENT_NAME-%J.out 
python ../../main.py \
  --model ../../../flask-experiments/qm9-r2-proponly-selfies-scratch-OOD_continue/logs/checkpoint-best-68000/ \
  --tokenizer ../../../flask-experiments/qm9-r2-proponly-selfies-scratch-OOD_continue/logs/checkpoint-best-68000/vocab.txt \
  --train-config ../../configs/train/qm9_r2_prop_only.json \
  --eval-only \
  --output-dir $MY_LOG_DIR \
  --eval-accumulation-steps 2 \
  --param-path ../../configs/eval/FLASK_qm9_r2_eval.json \
  --eval-file ../../OOD_data/qm9_r2_OOD/qm9_r2_OOD_ood_test.txt
