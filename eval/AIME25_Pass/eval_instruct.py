import hashlib
import json
import logging
import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

import lm_eval.models
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

from eval.task import BaseBenchmark

# Modified version of hendrycks_math with additional instruction to mark the solution with \\boxed
# https://github.com/mlfoundations/evalchemy/blob/e70a45e41cb2ada273d6bb98e75dba303ec31f8b/eval/chat_benchmarks/AIME25/eval_instruct.py#L15


SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user asks a question, and the assistant solves it. The "
    "assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think></think> tags, i.e., <think>\nThis is my "
    "reasoning.\n</think>\n\nYour final answer MUST BE put in \\boxed{}."
)

# Default k values for pass@k and majority voting@k
DEFAULT_K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]

# Default paths -- override via constructor arguments or environment variables
DEFAULT_DATA_FILE = os.environ.get(
    "AIME25_DATA_FILE",
    os.path.join(os.path.dirname(__file__), "data", "aime25.json"),
)
DEFAULT_CACHE_DIR = os.environ.get(
    "AIME25_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "denser", "eval"),
)


def estimate_pass_at_k(
    num_samples: int,
    num_correct: int,
    k: int,
) -> float:
    """Estimate pass@k using the unbiased estimator from the Codex paper.

    pass@k = 1 - C(n - c, k) / C(n, k)

    Uses the numerically stable version:
    1 - prod_{i=0}^{k-1} (n - c - i) / (n - i)

    Args:
        num_samples: Total number of generated samples (n)
        num_correct: Number of correct samples (c)
        k: Number of samples to consider

    Returns:
        Estimated pass@k probability
    """
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))


def majority_vote_at_k(
    answers: List[str],
    correct_answer: str,
    k: int,
    num_trials: int = 1000,
    rng: Optional[random.Random] = None,
) -> float:
    """Estimate majority voting@k accuracy via Monte Carlo sampling.

    Randomly samples k answers, takes the majority vote, and checks if it matches
    the correct answer. Repeats `num_trials` times and returns the fraction of
    trials where the majority vote was correct.

    Args:
        answers: List of all extracted model answers (strings)
        correct_answer: The ground-truth answer string
        k: Number of answers to sample for each vote
        num_trials: Number of Monte Carlo trials
        rng: Optional random.Random instance for reproducibility

    Returns:
        Estimated majority voting@k accuracy (fraction of trials correct)
    """
    if rng is None:
        rng = random.Random(42)

    if k == 1:
        # Special case: majority vote@1 == pass@1 (just pick one)
        correct_count = sum(1 for a in answers if is_equiv(str(correct_answer), a))
        return correct_count / len(answers)

    wins = 0
    for _ in range(num_trials):
        sampled = rng.sample(answers, min(k, len(answers)))
        counter = Counter(sampled)
        majority_answer = counter.most_common(1)[0][0]
        if is_equiv(str(correct_answer), majority_answer):
            wins += 1

    return wins / num_trials


def majority_vote_at_k_equiv(
    answers: List[str],
    correct_answer: str,
    k: int,
    num_trials: int = 1000,
    rng: Optional[random.Random] = None,
) -> float:
    """Majority voting@k with equivalence-aware grouping.

    Unlike the simple Counter-based version, this groups answers that are
    mathematically equivalent (via is_equiv) before taking the majority vote.

    Args:
        answers: List of all extracted model answers (strings)
        correct_answer: The ground-truth answer string
        k: Number of answers to sample for each vote
        num_trials: Number of Monte Carlo trials
        rng: Optional random.Random instance for reproducibility

    Returns:
        Estimated majority voting@k accuracy (fraction of trials correct)
    """
    if rng is None:
        rng = random.Random(42)

    # Pre-compute equivalence classes for all unique answers
    unique_answers = list(set(answers))
    equiv_classes: Dict[str, str] = {}
    representatives: List[str] = []

    for ans in unique_answers:
        found = False
        for rep in representatives:
            if is_equiv(ans, rep):
                equiv_classes[ans] = rep
                found = True
                break
        if not found:
            equiv_classes[ans] = ans
            representatives.append(ans)

    # Map all answers to their canonical form
    canonical_answers = [equiv_classes[a] for a in answers]

    # Check which canonical representative matches the correct answer
    correct_rep = None
    for rep in representatives:
        if is_equiv(str(correct_answer), rep):
            correct_rep = rep
            break

    if k == 1:
        correct_count = sum(1 for ca in canonical_answers if ca == correct_rep)
        return correct_count / len(canonical_answers)

    wins = 0
    for _ in range(num_trials):
        sampled = rng.sample(canonical_answers, min(k, len(canonical_answers)))
        counter = Counter(sampled)
        majority_canonical = counter.most_common(1)[0][0]
        if majority_canonical == correct_rep:
            wins += 1

    return wins / num_trials


class SolutionCache:
    """Disk-based cache for generated solutions.

    Stores solutions in a JSONL file keyed by a hash of the problem text.
    The cache file is located at:
        {cache_dir}/{task_name}/{safe_model_name}.jsonl

    Each line is a JSON object:
        {
            "question_id": <str>,
            "problem_hash": <str>,
            "model_name": <str>,
            "outputs": [<str>, ...],
            "model_answers": [<str>, ...],
        }

    Supports partial caching: if a question already has some solutions cached but fewer
    than num_samples, only the missing solutions are generated and appended.
    """

    def __init__(self, task_name: str, model_name: str, cache_dir: str = DEFAULT_CACHE_DIR):
        self.model_name = model_name
        self.task_name = task_name

        safe_model_name = model_name.replace("/", "_").replace("\\", "_")
        task_cache_dir = os.path.join(cache_dir, task_name)
        os.makedirs(task_cache_dir, exist_ok=True)

        self.cache_file = os.path.join(task_cache_dir, f"{safe_model_name}.jsonl")

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _problem_hash(self, problem: str) -> str:
        return hashlib.sha256(problem.encode()).hexdigest()[:16]

    def _load(self):
        """Load existing cache from disk."""
        if not os.path.exists(self.cache_file):
            return
        with open(self.cache_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                key = entry["problem_hash"]
                self._cache[key] = entry

    def _save(self):
        """Write full cache to disk (atomic via temp file)."""
        tmp_file = self.cache_file + ".tmp"
        with open(tmp_file, "w") as f:
            for entry in self._cache.values():
                f.write(json.dumps(entry) + "\n")
        os.replace(tmp_file, self.cache_file)

    def get_cached(self, problem: str) -> Optional[Dict[str, Any]]:
        """Get cached outputs for a problem, or None if not cached."""
        key = self._problem_hash(problem)
        return self._cache.get(key)

    def get_num_cached(self, problem: str) -> int:
        """Get the number of cached solutions for a problem."""
        entry = self.get_cached(problem)
        if entry is None:
            return 0
        return len(entry.get("outputs", []))

    def update(self, problem: str, question_id: str, outputs: List[str], model_answers: List[str]):
        """Update (or create) cache entry for a problem by appending new outputs."""
        key = self._problem_hash(problem)
        if key in self._cache:
            self._cache[key]["outputs"].extend(outputs)
            self._cache[key]["model_answers"].extend(model_answers)
        else:
            self._cache[key] = {
                "question_id": question_id,
                "problem_hash": key,
                "model_name": self.model_name,
                "outputs": list(outputs),
                "model_answers": list(model_answers),
            }

    def save(self):
        """Persist cache to disk."""
        self._save()


class AIME25Benchmark(BaseBenchmark):
    """
    AIME25 Benchmark for evaluating the math reasoning of LLMs.

    Follows the evaluation logic of hendrycks_math answer extraction.
    Generates multiple solutions per question and computes pass@k and majority voting@k.
    Supports disk-based caching to avoid re-generating already computed solutions.
    """

    def __init__(
        self,
        data_file: str = DEFAULT_DATA_FILE,
        cache_dir: str = DEFAULT_CACHE_DIR,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 2048,
        num_samples: int = 16,
        k_values: Optional[List[int]] = None,
        num_mv_trials: int = 1000,
        use_equiv_grouping: bool = True,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize AIME25 benchmark.

        Args:
            data_file: Path to the AIME25 dataset JSON file.
                       Override with AIME25_DATA_FILE env var.
            cache_dir: Root directory for the solution cache.
                       Override with AIME25_CACHE_DIR env var.
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility
            max_tokens: Maximum new tokens for generation
            num_samples: Number of solutions to generate per question (default: 16)
            k_values: List of k values for pass@k and majority voting@k.
                      Default: [1, 2, 4, 8, 16, 32, 64, 128]
            num_mv_trials: Number of Monte Carlo trials for majority voting estimation
            use_equiv_grouping: If True, group equivalent answers before majority voting
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file
        self.cache_dir = cache_dir
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.num_samples = num_samples
        self.k_values = k_values or DEFAULT_K_VALUES
        self.num_mv_trials = num_mv_trials
        self.use_equiv_grouping = use_equiv_grouping
        self.task_name = "AIME25_Pass"

        # Filter k_values to only include those <= num_samples
        self.k_values = [k for k in self.k_values if k <= self.num_samples]

    def _get_model_name(self, model: LM) -> str:
        """Extract a canonical model name string."""
        if isinstance(model, lm_eval.models.huggingface.HFLM):
            return model.pretrained
        elif isinstance(model, lm_eval.models.openai_completions.OpenAIChatCompletion):
            return f"openai/{model.model}"
        else:
            return model.model_args["model"]

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate multiple solution completions per question using the provided model.
        Uses disk-based caching to skip already-generated solutions.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and metadata,
            or None for non-primary ranks
        """
        examples = self.load_questions()
        model_name = self._get_model_name(model)

        # Set up cache
        cache = SolutionCache(self.task_name, model_name, cache_dir=self.cache_dir)
        cached_total = sum(cache.get_num_cached(ex["problem"]) for ex in examples)
        self.logger.info(
            f"Cache loaded from {cache.cache_file}: "
            f"{cached_total} total cached solutions across {len(examples)} questions"
        )

        # Determine how many new samples are needed per question
        samples_needed = []
        samples_cached = []
        for example in examples:
            n_cached = cache.get_num_cached(example["problem"])
            n_needed = max(0, self.num_samples - n_cached)
            samples_needed.append(n_needed)
            samples_cached.append(n_cached)

        total_needed = sum(samples_needed)
        total_cached = self.num_samples * len(examples) - total_needed
        self.logger.info(
            f"Solutions needed: {total_needed} new + {total_cached} cached "
            f"= {self.num_samples * len(examples)} total "
            f"({len(examples)} questions x {self.num_samples} samples)"
        )

        # Build instances only for uncached samples.
        # IMPORTANT: Each sample gets a unique seed so that lm_eval does not
        # deduplicate identical instances and return the same output N times.
        # The seed offset accounts for already-cached samples so that resuming
        # a partial run produces new distinct samples rather than re-generating
        # the same ones.
        all_instances = []
        # Track which question index and how many samples each instance block belongs to
        instance_question_map = []  # list of (question_idx, n_needed)

        instance_idx = 0
        for q_idx, (example, n_needed, n_cached) in enumerate(
            zip(examples, samples_needed, samples_cached)
        ):
            if n_needed == 0:
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ]
            templated_messages = self._prepare_messages(messages, model)

            instance_question_map.append((q_idx, n_needed))

            for sample_idx in range(n_cached, n_cached + n_needed):
                # Unique seed per sample: vary the first element of the seed list
                # so each Instance has distinct generation kwargs, preventing dedup.
                # The seed must remain a list since the framework indexes it with seeds[0].
                sample_seed = [self.seed[0] + sample_idx] + self.seed[1:]
                all_instances.append(
                    Instance(
                        "generate_until",
                        example,
                        (
                            templated_messages,
                            {
                                "do_sample": True,
                                "max_new_tokens": self.max_new_tokens,
                                "temperature": 0.7,
                                "seed": sample_seed,
                            },
                        ),
                        instance_idx,
                    )
                )
                instance_idx += 1

        # Generate model responses (only for uncached)
        if all_instances:
            self.logger.info(f"Generating {len(all_instances)} new solutions...")
            outputs = self.compute(model, all_instances)
        else:
            self.logger.info("All solutions are cached. Skipping generation.")
            outputs = []

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        # Merge new outputs into cache
        output_offset = 0
        for q_idx, n_needed in instance_question_map:
            example = examples[q_idx]
            new_outputs = outputs[output_offset : output_offset + n_needed]
            new_answers = [self.extract_answer(o) for o in new_outputs]
            output_offset += n_needed

            question_id = example.get("id", example.get("unique_id", str(q_idx)))
            cache.update(example["problem"], str(question_id), new_outputs, new_answers)

        # Save cache to disk
        cache.save()
        self.logger.info(f"Cache saved to {cache.cache_file}")

        # Assemble final per-question data from cache
        for q_idx, example in enumerate(examples):
            cached_entry = cache.get_cached(example["problem"])
            all_outputs = cached_entry["outputs"][: self.num_samples]
            all_answers = cached_entry["model_answers"][: self.num_samples]
            example["model_outputs"] = all_outputs
            example["model_answers"] = all_answers
            example["sample_correct"] = [
                is_equiv(str(example["answer"]), ans) for ans in all_answers
            ]

        return {"examples": examples, "num_samples": self.num_samples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions with pass@k and majority voting@k."""

        # Handle None result from non-primary ranks
        if results is None:
            return None

        examples = results["examples"]
        num_samples = results["num_samples"]
        total = len(examples)

        rng = random.Random(42)

        mv_func = majority_vote_at_k_equiv if self.use_equiv_grouping else majority_vote_at_k

        # Compute per-question metrics
        pass_at_k_per_question = {k: [] for k in self.k_values}
        mv_at_k_per_question = {k: [] for k in self.k_values}

        for example in examples:
            num_correct = sum(example["sample_correct"])
            answers = example["model_answers"]
            correct_answer = example["answer"]

            for k in self.k_values:
                # pass@k
                p_at_k = estimate_pass_at_k(num_samples, num_correct, k)
                pass_at_k_per_question[k].append(p_at_k)

                # majority voting@k
                mv_at_k = mv_func(
                    answers, correct_answer, k,
                    num_trials=self.num_mv_trials, rng=rng,
                )
                mv_at_k_per_question[k].append(mv_at_k)

            # Store per-question summary
            example["num_correct"] = num_correct
            example["num_samples"] = num_samples

        # Aggregate metrics
        metrics = {
            "num_total": total,
            "num_samples_per_question": num_samples,
        }

        for k in self.k_values:
            metrics[f"pass@{k}"] = float(np.mean(pass_at_k_per_question[k]))
            metrics[f"majority_vote@{k}"] = float(np.mean(mv_at_k_per_question[k]))

        # Also include greedy-equivalent accuracy (pass@1 is the same)
        metrics["accuracy"] = metrics["pass@1"]

        self.logger.info("=== AIME25 Results ===")
        for k in self.k_values:
            self.logger.info(f"  pass@{k}: {metrics[f'pass@{k}']:.4f}")
            self.logger.info(f"  majority_vote@{k}: {metrics[f'majority_vote@{k}']:.4f}")

        results.update(metrics)
        return results

    def load_questions(self) -> List[Dict[str, str]]:
        """Load AIME25 questions from the data file."""
        with open(self.data_file, "r") as f:
            questions = [json.loads(x) for x in f]

        if self.debug:
            questions = questions[:2]
            self.logger.info(f"Debug mode enabled. Using only {len(questions)} questions.")

        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        return questions

    def extract_answer(self, output: str) -> str:
        """Extract the final answer from a model-generated solution.

        Expected format: \\boxed{answer}

        Uses the same logic as hendrycks_math.

        Args:
            output (str): Model-generated solution text

        Returns:
            str: Extracted final answer. Returns empty string if no answer found.
        """
        try:
            answer = remove_boxed(last_boxed_only_string(output))
            return answer
        except:
            return ""