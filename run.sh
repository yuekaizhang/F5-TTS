export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/workspace/flow_rl/F5-TTS/src:$PYTHONPATH
# python src/f5_tts/train/datasets/prepare_wenetspeech4tts.py


# accelerate config


# Pretraining phase
# accelerate launch src/f5_tts/train/train.py

# # GRPO phase
# accelerate launch src/f5_tts/train/train_rl.py