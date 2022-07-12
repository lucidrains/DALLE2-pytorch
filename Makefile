install:
	pip install -U pip
	pip install -e .

test_decoder:
	CUDA_VISIBLE_DEVICES= python train_decoder.py --config_file configs/train_decoder_config.test.json

test_upsampler:
	CUDA_VISIBLE_DEVICES= python train_decoder.py --config_file configs/train_upsampler_config.test.json

test: test_decoder test_upsampler
	echo it works
