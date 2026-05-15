# Intentionally minimal package init.
#
# Earlier this module eagerly imported ``rl_training_validation.utils.multi_task_env``,
# which pulled in ``multiros.wrappers.*`` at import time. That broke any
# tool that wanted to introspect this package without a sourced ROS
# workspace (audit scripts, env_safety helpers, IDE indexers).
#
# Each consumer that actually needs ``multi_task_env`` imports it
# explicitly:
#
#     from rl_training_validation.utils.multi_task_env import MultiTaskEnv
