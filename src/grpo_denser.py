# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "Pillow",
#     "peft",
#     "math-verify",
#     "latex2sympy2_extended",
#     "torchvision",
#     "trackio",
#     "kernels",
# ]
# ///

"""
GRPO training with Divergence-Enhanced Nuanced Supervision for Effective Reasoning (DENSER).

DENSER focuses gradient on the tokens where a rollout's hidden representations
diverge from the opposite reward class, and upweights rollouts that represent
unique solution strategies within their class.

Usage:
    accelerate launch \
        --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
        GRPO_LLM.py \
        --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
        --output_dir grpo-denser-Qwen2.5-VL-3B-Instruct \
        --learning_rate 1e-5 \
        --gradient_checkpointing \
        --dtype bfloat16 \
        --max_prompt_length 2048 \
        --max_completion_length 1024 \
        --use_vllm \
        --vllm_mode colocate \
        --use_peft \
        --lora_target_modules "q_proj", "v_proj" \
        --log_completions
"""

import os

import torch
from datasets import load_dataset
from dataclasses import dataclass, field

from trl import (
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.rewards import accuracy_reward, think_format_reward
from grpo_denser_trainer import GRPODENSERTrainer


@dataclass
class DENSERConfig(GRPOConfig):
    """GRPOConfig extended with DENSER hyperparameters."""

    use_denser: bool = field(
        default=True,
        metadata={"help": "Enable divergence-enhanced nuanced supervision for effective reasoning."},
    )
    denser_alpha_cross: float = field(
        default=1.0,
        metadata={
            "help": (
                "Scaling factor for cross-class divergence signal. "
                "0 = ignore cross-class, higher = more weight on "
                "tokens that differ from the opposite reward class."
            )
        },
    )
    denser_alpha_within: float = field(
        default=1.0,
        metadata={
            "help": (
                "Scaling factor for within-class uniqueness signal. "
                "0 = ignore uniqueness, higher = more diversity-seeking "
                "(upweights rare solution strategies)."
            )
        },
    )
    denser_beta: float = field(
        default=1.0,
        metadata={
            "help": (
                "Blending strength between uniform weights (0.0) and "
                "DENSER weights (1.0). Controls how much token-level "
                "redistribution DENSER applies. Try 0.1-0.3 for a gentle nudge."
            )
        },
    )
    denser_window_size: int = field(
        default=5,
        metadata={"help": "Proportional alignment half-window for token matching."},
    )


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DENSERConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # Model
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config

    ################
    # Prompt
    ################
    SYSTEM_PROMPT = (
        "A conversation between user and assistant. The user asks a question, and the assistant solves it. The "
        "assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think></think> tags, i.e., <think>\nThis is my "
        "reasoning.\n</think>\n\nYour final answer MUST BE put in \\boxed{}."
    )

    def make_conversation(example):
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"] if "problem" in example else example["prompt"]},
        ]
        return {"prompt": prompt}

    ################
    # Dataset
    ################
    train_dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    train_dataset = train_dataset.select(range(1000))
    train_dataset = train_dataset.map(make_conversation)

    eval_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    eval_dataset = eval_dataset.select(range(100))
    eval_dataset = eval_dataset.map(make_conversation)
    eval_dataset = eval_dataset.remove_columns(["problem"])

    ################
    # Training
    ################
    trainer = GRPODENSERTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[accuracy_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )
    trainer.tools = None
    trainer.vllm_generation.tools = None
    
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)