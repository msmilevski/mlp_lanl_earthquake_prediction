python scripts/experiments/lstm_experiment.py --data_path /home/dziugas/lanl_earth/data \
											 --experiment_name "lstm_overlapped_is_baseline" \
											 --segment_size 150000 --element_size 1000 \
											 --use_gpu "true" \
											 --num_epochs 100 --dropout 0 \
											 --learning_rate 0.0002 --batch_size 1 \
											 --num_layers 2