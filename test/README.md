### Playing with π₀ / π₀.₅ models for a deep understanding. 

We are going to make inference from Datasets (No Robot)


1. **Install with π support**
   - From this repo root:
     ```bash
     python -m venv .venv
     source .venv/bin/activate  # or .venv\Scripts\activate on Windows
     pip install --upgrade pip
     pip install -e ".[pi]"
     ```

2. **Choose a LeRobot dataset**
   - 

3. **Load dataset metadata**
   - Minimal example:
     ```python
     from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

     ds_meta = LeRobotDatasetMetadata("lerobot/your_dataset_id")
     print(ds_meta.features.keys())  # observation.* and action.* keys
     ```

4. **Instantiate a π policy and processors**
   - Example for π₀:
     ```python
     import torch
     from lerobot.configs.policies import PreTrainedConfig
     from lerobot.policies.factory import make_policy, make_pi0_pre_post_processors
     from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

     device = torch.device(
         "cuda" if torch.cuda.is_available()
         else "mps" if torch.backends.mps.is_available()
         else "cpu"
     )

     ds_meta = LeRobotDatasetMetadata("lerobot/your_dataset_id")

     cfg = PreTrainedConfig.from_dict({
         "type": "pi0",                     # or "pi05"
         "pretrained_path": "lerobot/pi0_base",
         "device": str(device),
     })

     policy = make_policy(cfg, ds_meta=ds_meta)
     preproc, postproc = make_pre_post_processors(
         policy.config,
         pretrained_path=cfg.pretrained_path,
         dataset_stats=ds_meta.stats,
     )
     ```

5. **Run offline inference on dataset frames**
   - Pseudo-code sketch:
     ```python
     from lerobot.datasets.lerobot_dataset import LeRobotDataset

     dataset = LeRobotDataset("lerobot/your_dataset_id", episodes=[0])

     for frame in dataset:
         # frame is a dict of tensors / arrays matching ds_meta.features
         obs = {
             k: v for k, v in frame.items()
             if k.startswith("observation.")
         }

         # Add any complementary data expected by π policies, e.g. task string
         obs["task"] = "pick up the red block"

         policy_input = preproc(obs)
         action = policy.select_action(policy_input)
         action = postproc(action)

         # Here you can log, visualize, or compare `action` to the dataset's recorded actions
     ```

This workflow lets you experiment with π₀ / π₀.₅ behavior purely from logged data (images + states + actions), without needing a real robot or simulator.