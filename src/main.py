import torch

from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# first we need the dataset, or directly input. robot
# robot

device = "cuda"

REPO_ID = "NONHUMAN-RESEARCH/TEST_RECORD_ANNOTATIONS"
PI0_MODEL_ID = "lerobot/pi0_base"

steps = 2

policy = PI0Policy.from_pretrained(PI0_MODEL_ID)
policy.config.device = str(device)
policy.to(device)
policy.eval()

preproc, postproc = make_pi0_pre_post_processors(
    config=policy.config,
    dataset_stats=None,
)


def build_batch_from_frame(frame, task, index):
    batch = {key: value for key, value in frame.items() if key.startswith("observation.")}
    batch["task"] = task
    batch["index"] = index
    return batch


def main():
    dataset = LeRobotDataset(REPO_ID)
    task = "Wave your hand"
    for i in range(steps):
        # Frame is a dict, the keys in you can find in the meta/info.json, features keys.
        frame = dataset[i]

        batch = build_batch_from_frame(frame, task, i)
        # The problem is that pi zero expects another names for the keys.
        # Now the prepoc, what it expect? I mean the images, the prompt, right?
        policy_input = preproc(batch)
        with torch.no_grad():
            policy_action = policy.select_action(policy_input)
            policy_action = postproc(policy_action)

        print(policy_action)


if __name__ == "__main__":
    main()
