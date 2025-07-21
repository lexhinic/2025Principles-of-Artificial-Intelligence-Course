python main.py \
  --data_dir ./garbage-dataset \
  --model_type basic \
  --epochs 200 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --weight_decay 0.0001 \
  --optimizer adam \
  --save_dir ./garbage-classification-basic-aug \
  --data_augmentation \