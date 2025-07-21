#!/bin/bash
# filepath: /home/stu4/garbage_classification/scripts/hyperparameter_search.sh

LOG_DIR="experiments_logs"
mkdir -p $LOG_DIR

# 记录实验开始时间
echo "开始超参数搜索: $(date)" | tee $LOG_DIR/experiment_summary.log

# 定义要测试的超参数
MODEL_TYPES=("basic" "resnet" "improved")
BATCH_SIZES=(16 32 64)
LEARNING_RATES=(0.01 0.001 0.0001)
WEIGHT_DECAYS=(0.0001 0.00001)
OPTIMIZERS=("sgd" "adam")
DATA_AUG=(true false)  # 是否使用数据增强
EPOCHS=20  

# 设置数据目录
DATA_DIR="./garbage-dataset"

# 记录实验配置
echo "实验配置:" | tee -a $LOG_DIR/experiment_summary.log
echo "模型类型: ${MODEL_TYPES[*]}" | tee -a $LOG_DIR/experiment_summary.log
echo "批次大小: ${BATCH_SIZES[*]}" | tee -a $LOG_DIR/experiment_summary.log
echo "学习率: ${LEARNING_RATES[*]}" | tee -a $LOG_DIR/experiment_summary.log
echo "权重衰减: ${WEIGHT_DECAYS[*]}" | tee -a $LOG_DIR/experiment_summary.log
echo "优化器: ${OPTIMIZERS[*]}" | tee -a $LOG_DIR/experiment_summary.log
echo "数据增强: ${DATA_AUG[*]}" | tee -a $LOG_DIR/experiment_summary.log
echo "训练轮数: $EPOCHS" | tee -a $LOG_DIR/experiment_summary.log
echo "----------------------------------------" | tee -a $LOG_DIR/experiment_summary.log

# 存储最佳配置
BEST_ACC=0
BEST_CONFIG=""

# 遍历所有超参数组合
for model in "${MODEL_TYPES[@]}"; do
  for batch_size in "${BATCH_SIZES[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      for wd in "${WEIGHT_DECAYS[@]}"; do
        for opt in "${OPTIMIZERS[@]}"; do
          for aug in "${DATA_AUG[@]}"; do
            # 创建实验ID和保存目录
            EXP_ID="${model}_bs${batch_size}_lr${lr}_wd${wd}_${opt}"
            if [ "$aug" = true ]; then
              EXP_ID="${EXP_ID}_aug"
              AUG_FLAG="--data_augmentation"
            else
              AUG_FLAG=""
            fi
            
            SAVE_DIR="./checkpoints/${EXP_ID}"
            mkdir -p $SAVE_DIR
            
            # 记录当前实验
            echo "运行实验: $EXP_ID" | tee -a $LOG_DIR/experiment_summary.log
            
            # 构建命令
            CMD="python main.py \
              --data_dir $DATA_DIR \
              --model_type $model \
              --batch_size $batch_size \
              --learning_rate $lr \
              --weight_decay $wd \
              --optimizer $opt \
              --epochs $EPOCHS \
              --save_dir $SAVE_DIR \
              $AUG_FLAG"
            
            echo "命令: $CMD" | tee -a $LOG_DIR/experiment_summary.log
            
            LOG_FILE="$LOG_DIR/${EXP_ID}.log"
            echo "开始实验 $EXP_ID: $(date)" | tee -a $LOG_FILE
            
            eval $CMD | tee -a $LOG_FILE
            TEST_ACC=$(grep -o "Accuracy: [0-9.]\+" $LOG_FILE | tail -1 | cut -d' ' -f2)
            
            echo "实验 $EXP_ID 完成，测试准确率: $TEST_ACC" | tee -a $LOG_DIR/experiment_summary.log
            
            # 更新最佳配置
            if (( $(echo "$TEST_ACC > $BEST_ACC" | bc -l) )); then
              BEST_ACC=$TEST_ACC
              BEST_CONFIG=$EXP_ID
            fi
            
            echo "----------------------------------------" | tee -a $LOG_DIR/experiment_summary.log
          done
        done
      done
    done
  done
done

# 输出最佳配置
echo "超参数搜索完成！" | tee -a $LOG_DIR/experiment_summary.log
echo "最佳配置: $BEST_CONFIG" | tee -a $LOG_DIR/experiment_summary.log
echo "最佳准确率: $BEST_ACC" | tee -a $LOG_DIR/experiment_summary.log
echo "实验结束时间: $(date)" | tee -a $LOG_DIR/experiment_summary.log

# 将最佳模型复制到根目录
cp -r "./checkpoints/${BEST_CONFIG}/best_model.pth" "./best_model.pth"
echo "最佳模型已复制到 ./best_model.pth"