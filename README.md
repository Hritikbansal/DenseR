# DenseR
Dense Reward For Free in LLM Reasoning

This repo contains the code for the blog post: https://huggingface.co/blog/hbXNov/denser.

## Instructions

The code is based on TRL repo: https://github.com/huggingface/trl

Please follow their instructions for installation and environment setup.

## Scripts

The scripts are hosted in the [src](src/) folder.

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file ds_config/config.yaml \
    src/grpo_denser.py \
    --model_name_or_path Qwen/Qwen3-0.6B-Base \
    --output_dir output/qwen3_0.6b_grpo_denser \
    --dtype float16 \
    --learning_rate 1e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --max_completion_length 2048 \
    --temperature 0.7 \
    # --- Generation ---
    --use_vllm \
    --vllm_mode colocate \
    --vllm_importance_sampling_correction \
    --vllm_max_model_length 2560 \
    --num_generations 8 \
    --num_generations_eval 1 \
    # --- GRPO ---
    --beta 0.0 \
    --epsilon_high 0.28 \
    # --- DENSER ---
    --use_denser True \
    --denser_alpha_cross 1.0 \
    --denser_alpha_within 0.3 \
    --denser_window_size 5 \
    --denser_beta 0.1 \
    # --- Logging & Saving ---
    --logging_steps 10 \
    --log_completions \
    --do_eval \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 0.2 \
    --save_only_model True \
    --run_name qwen3_0.6b_grpo_denser
```

## Evaluation

I use [evalchemy](https://github.com/mlfoundations/evalchemy.git) repo to evaluate the trained models.

Please refer to their code for installation. 

I provide the folder to get 16 samples per question from AIME25 and computing pass@k and mv@k metrics. Please add this folder in the [evalchemy codebase](https://github.com/mlfoundations/evalchemy/tree/main/eval/chat_benchmarks) to run it.

```bash
DEFAULT_DATA_FILE=[path to aime25 json file] \
DEFAULT_CACHE_DIR=[path to cache dir] \
VLLM_ATTENTION_BACKEND=XFORMERS \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
CUDA_VISIBLE_DEVICES=0 \
python -m eval.eval \
    --model vllm \
    --tasks "AIME25_Pass" \
    --model_args "pretrained=<path_to_checkpoint>" \
    --batch_size 8 \
    --output_path logs
```