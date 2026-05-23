# Minimal package init: avoid eager imports that require a sourced ROS
# workspace, so introspection tools (env_safety helpers, IDE indexers)
# work without one. Consumers that need ``multi_task_env`` import it
# explicitly:
#
#     from rl_training_validation.utils.multi_task_env import MultiTaskEnv
