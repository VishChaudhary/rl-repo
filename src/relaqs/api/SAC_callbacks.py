from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
import torch
from torch.linalg import vector_norm
from typing import Dict, Tuple

class SACGateSynthesisCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        worker.env.episode_id = episode.episode_id

        episode.hist_data["q_values"] = []  # For Q-value tracking (from both Q-networks)
        episode.hist_data["grad_gnorm"] = []  # Gradient norm of updates
        episode.hist_data["average_gradnorm"] = []  # Average gradient norm
        episode.hist_data["actions"] = []  # Actions taken during the episode

    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, Tuple[Policy, SampleBatch]],
        **kwargs
    ):
        alg_config = worker.policy_map["default_policy"].config
        verbose = alg_config["env_config"]["verbose"]

        if verbose:
            print("-------------------Post-Processing Batch-------------------")

        # Initialize a counter for batches if it doesn't exist
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1

        # Get the policy and model
        policy = worker.get_policy("default_policy")
        model = policy.model

        # --------------------> Q-Value Tracking (Twin Q-Networks) <--------------------
        input_dict = SampleBatch(obs=torch.Tensor(postprocessed_batch['obs']))
        model_out_t, _ = model(input_dict, [], None)
        q_values = model.get_q_values(model_out_t, torch.Tensor(postprocessed_batch['actions']))

        # Unpack Q-values (SAC uses twin Q-networks, returning q1 and q2)
        q1, q2 = q_values  # Unpack the tuple
        # print(f"Type of q_values: {type(q_values)}")
        # print(f"Contents of q_values: {q_values}")
        # print(f"Q1 values: {q1}")
        # print(f"Q2 values: {q2}")

        # Detach both Q-values and convert to numpy
        q1_np = q1.detach().numpy()
        # print(f"Q1 values: {q1_np}")
        # q2_np = q2.detach().numpy()

        # Append both Q-values to the hist_data for tracking
        episode.hist_data["q_values"].append(q1_np)

        # --------------------> Gradient Norm Tracking <--------------------
        batch = SampleBatch(
            obs=torch.Tensor(postprocessed_batch['obs']),
            actions=torch.Tensor(postprocessed_batch['actions']),
            new_obs=torch.Tensor(postprocessed_batch['new_obs']),
            rewards=torch.Tensor(postprocessed_batch['rewards']),
            terminateds=torch.Tensor(postprocessed_batch['terminateds']),
            truncateds=torch.Tensor(postprocessed_batch['truncateds']),
            weights=torch.Tensor(postprocessed_batch['weights']),
        )

        gradients = policy.compute_gradients(batch)
        gradients_info = gradients[1]
        NoneType = type(None)
        gradients = [x for x in gradients[0] if not isinstance(x, NoneType)]
        average_grad = 0
        for grad in gradients:
            average_grad += vector_norm(grad)
        average_grad = average_grad/(len(gradients))

        episode.hist_data['grad_gnorm'].append(gradients_info['learner_stats']['grad_gnorm'])
        episode.hist_data["average_gradnorm"].append(average_grad.numpy())

        # --------------------> Action Tracking <--------------------
        episode.hist_data["actions"].append(postprocessed_batch["actions"].tolist())

    # You can uncomment this if you need to track custom metrics at episode end.
    # def on_episode_end(
    #     self,
    #     *,
    #     worker: RolloutWorker,
    #     base_env: BaseEnv,
    #     policies: Dict[str, Policy],
    #     episode: Episode,
    #     env_index: int,
    #     **kwargs
    # ):
    #     episode.custom_metrics["actions"] = episode.user_data
