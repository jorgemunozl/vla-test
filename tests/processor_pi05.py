"""
Processor for the PI05 policy considering the subtask prediction
task frequency and the fast tokenizer for the training.
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from xhuman.policies.pi05.configuration_pi05 import PI05Config
from xhuman.policies.pi05.modeling_pi_05 import pad_vector

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)

from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
    create_transition,
    _extract_complementary_data,
)
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    OBS_PREFIX,
    ACTION,
    REWARD,
    DONE,
    TRUNCATED,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from transformers import AutoTokenizer


def batch_to_transition_with_time_index(
    batch: dict[str, Any]
) -> EnvTransition:
    """
    Custom batch_to_transition converter that includes 'frame_index' in
    complementary_data.

    This extends the default lerobot batch_to_transition to include
    'frame_index' and 'subtask' keys in the complementary_data dictionary.

    Args:
        batch: A batch dictionary that may contain 'frame_index' and
            'subtask' keys.

    Returns:
        An EnvTransition dictionary with frame_index and subtask in
        complementary_data.
    """
    # Use the default extraction but add frame_index and subtask
    complementary_data = _extract_complementary_data(batch)

    if "frame_index" in batch:
        complementary_data["frame_index"] = batch["frame_index"]

    # Add subtask if present in batch
    if "subtask" in batch:
        complementary_data["subtask"] = batch["subtask"]

    # Extract observation keys (same as default)
    observation_keys = {
        k: v for k, v in batch.items() if k.startswith(OBS_PREFIX)
    }

    return create_transition(
        observation=observation_keys if observation_keys else None,
        action=batch.get(ACTION),
        reward=batch.get(REWARD, 0.0),
        done=batch.get(DONE, False),
        truncated=batch.get(TRUNCATED, False),
        info=batch.get("info", {}),
        complementary_data=(
            complementary_data if complementary_data else None
        ),
    )


@ProcessorStepRegistry.register(
    name="pi05_ki_prepare_state_tokenizer_processor_step"
)
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    Supports both subtask generation and action generation formats.

    Flow:
    1. When time_index % subtask_prediction_frequency == 0:
       Generate subtask with prompt: "Task: {task}. Subtask: "
    2. Otherwise:
       Generate actions with prompt:
       "Task: {cached_subtask}, State: {state};\\nAction: "

    The subtask is cached and reused until a new one is generated.
    """

    max_state_dim: int = 32
    task_key: str = "task"
    subtask_key: str = "subtask"
    subtask_prediction_frequency: int = 100
    frame_index_key: str = "frame_index"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        import string
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(
            TransitionKey.COMPLEMENTARY_DATA, {}
        ).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # Check if we should generate subtask based on time_index
        complementary_data = transition.get(
            TransitionKey.COMPLEMENTARY_DATA, {}
        )
        time_index = complementary_data.get(self.frame_index_key, None)

        generate_subtask = (
            self.subtask_prediction_frequency > 0
            and time_index is not None
            and time_index % self.subtask_prediction_frequency == 0
        )
        # Get cached subtask for action generation
        cached_subtask = complementary_data.get(self.subtask_key, None)

        state = deepcopy(state)

        # Prepare state (pad to max_state_dim)
        state = pad_vector(state, self.max_state_dim)

        # State should already be normalized to [-1, 1] by the
        # NormalizerProcessorStep that runs before this step
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(
            state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]
        ) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            cleaned_text_lower = cleaned_text.lower()
            # Remove last punctuation if present and add period
            if (
                cleaned_text_lower
                and cleaned_text_lower[-1] in string.punctuation
            ):
                cleaned_text_lower = cleaned_text_lower[:-1]
            cleaned_text_lower += '.'

            if generate_subtask:
                # Format for subtask generation: "Task: {task}. Subtask: "
                # Model will generate the subtask autoregressively
                full_prompt = f"Task: {cleaned_text_lower} Subtask: "
            else:
                # Format for action generation with subtask context:
                state_str = " ".join(map(str, discretized_states[i]))

                if cached_subtask is not None:
                    # Clean up subtask text
                    subtask_text = cached_subtask
                    if isinstance(subtask_text, list):
                        subtask_text = subtask_text[i] if i < len(
                            subtask_text
                        ) else subtask_text[0]
                    subtask_text = str(subtask_text).strip()
                    # Remove trailing punctuation and add period
                    if (
                        subtask_text
                        and subtask_text[-1] in string.punctuation
                    ):
                        subtask_text = subtask_text[:-1]
                    subtask_text += '.'
                    full_prompt = (
                        f"Task: {subtask_text} "
                        f"State: {state_str};\nAction: "
                    )
                else:
                    # No subtask available yet, use task only with state
                    full_prompt = (
                        f"Task: {cleaned_text}, "
                        f"State: {state_str};\nAction: "
                    )
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][
            self.task_key
        ] = full_prompts
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


@ProcessorStepRegistry.register(
    name="pi05_ki_detokenize_subtask_processor_step"
)
@dataclass
class Pi05DetokenizeSubtaskProcessorStep(ProcessorStep):
    """
    Processor step to detokenize subtask tokens if they are present.
    Only detokenizes when subtask_tokens is not None.
    """

    tokenizer_name: str = "google/paligemma-3b-pt-224"
    subtask_tokens_key: str = "subtask_tokens"
    subtask_key: str = "subtask"

    def __post_init__(self):
        """Initialize tokenizer lazily on first use."""
        self._tokenizer = None

    def _get_tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name
            )
        return self._tokenizer

    def __call__(self, action: PolicyAction) -> PolicyAction:
        """
        Detokenize subtask tokens if present.

        Args:
            action: PolicyAction (tensor or dict) that may contain
                'subtask_tokens' key

        Returns:
            PolicyAction with 'subtask' key added if subtask_tokens
            were present. If input was a tensor and no subtask_tokens,
            returns tensor as-is.
        """
        # PolicyAction can be either a tensor or a dict
        # If it's a tensor, return as-is (no subtask tokens to detokenize)
        if not isinstance(action, dict):
            return action

        subtask_tokens = action.get(self.subtask_tokens_key)

        # Only detokenize if subtask_tokens is not None
        if subtask_tokens is not None:
            tokenizer = self._get_tokenizer()

            # Handle different tensor formats
            if isinstance(subtask_tokens, torch.Tensor):
                # Convert to numpy/cpu if needed
                if subtask_tokens.is_cuda:
                    subtask_tokens = subtask_tokens.cpu()
                subtask_tokens = subtask_tokens.numpy()

            # Convert to list of lists if needed
            if isinstance(subtask_tokens, np.ndarray):
                subtask_tokens = subtask_tokens.tolist()

            # Handle empty or invalid subtask_tokens
            if not subtask_tokens or len(subtask_tokens) == 0:
                action[self.subtask_key] = None
                return action

            # Detokenize each sequence in the batch
            # subtask_tokens shape: (B, max_decoding_steps) or
            # (max_decoding_steps,)
            if isinstance(subtask_tokens[0], (list, np.ndarray)):
                # Batch case: (B, max_decoding_steps)
                detokenized_subtasks = []
                for token_seq in subtask_tokens:
                    # Remove padding tokens (0) and EOS tokens (1)
                    # Filter out special tokens
                    filtered_tokens = [
                        int(token)
                        for token in token_seq
                        if (
                            int(token) != 0
                            and int(token) != 1
                            and int(token) < tokenizer.vocab_size
                        )
                    ]
                    if filtered_tokens:
                        detokenized = tokenizer.decode(
                            filtered_tokens, skip_special_tokens=True
                        )
                        detokenized_subtasks.append(detokenized.strip())
                    else:
                        detokenized_subtasks.append(None)

                # If batch size is 1, return single string
                if len(detokenized_subtasks) == 1:
                    action[self.subtask_key] = detokenized_subtasks[0]
                else:
                    action[self.subtask_key] = detokenized_subtasks
            else:
                # Single sequence case: (max_decoding_steps,)
                filtered_tokens = [
                    int(token)
                    for token in subtask_tokens
                    if (
                        int(token) != 0
                        and int(token) != 1
                        and int(token) < tokenizer.vocab_size
                    )
                ]
                if filtered_tokens:
                    detokenized = tokenizer.decode(
                        filtered_tokens, skip_special_tokens=True
                    )
                    action[self.subtask_key] = detokenized.strip()
                else:
                    action[self.subtask_key] = None

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


def make_pi05_pre_post_processors_ki(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for
       tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.
    3. Detokenizing subtask tokens if present.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and
        post-processor pipelines.
    """

    # Add remaining processors
    input_steps: list[ProcessorStep] = [
        # To mimic the same processor as pretrained one
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        # NOTE: NormalizerProcessorStep MUST come before
        # Pi05PrepareStateTokenizerProcessorStep because the tokenizer step
        # expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(
            max_state_dim=config.max_state_dim,
            subtask_prediction_frequency=config.subtask_prediction_frequency,
        ),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
        Pi05DetokenizeSubtaskProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
        ),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=batch_to_transition_with_time_index,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
