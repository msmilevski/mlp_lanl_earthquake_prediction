python scripts/experiments/lstm_experiment.py --data_path ./data \
											 --experiment_name "lstm_new_overlapped_provider" \
											 --segment_size 150000 --element_size 1000 \
											 --use_gpu "true" --gpu_id "0,1" \
											 --num_epochs 100 --dropout 0 \
											 --learning_rate 0.001 --batch_size 64 \
											 --num_layers 3 --overlapped_data "true" --overlap_fraction 0.9
