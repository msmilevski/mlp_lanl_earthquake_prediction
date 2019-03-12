python scripts/experiments/lstm_experiment.py --data_path ./data \
											 --experiment_name "lstm_overlapped_batch_15" \
											 --segment_size 150000 --element_size 1000 \
											 --use_gpu "true" \
											 --num_epochs 100 --dropout 0.1 \
											 --learning_rate 0.0002 --batch_size 15 \
											 --num_layers 3 --overlapped_data "true"
