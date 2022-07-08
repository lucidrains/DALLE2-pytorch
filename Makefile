install:
	pip install -U pip
	pip install -e .

test:
	CUDA_VISIBLE_DEVICES= python train_decoder.py --config_file configs/train_decoder_config.test.json
