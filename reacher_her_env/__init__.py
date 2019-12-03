"""Register new 2 Link robot arm"""
from gym.envs.registration import register

register(
    id='ReacherHerEnv-v2',
    entry_point='reacher_her_env.reacher_env_mod:ReacherHerEnv',
    max_episode_steps=150,
    reward_threshold=18.0,
)