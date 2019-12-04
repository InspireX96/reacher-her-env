"""
Reacher HER Env
"""

import numpy as np
from gym.envs.mujoco.reacher import ReacherEnv


class ReacherHerEnv(ReacherEnv):
    """
    Reacher HER env

    :param ReacherEnv: obj, original gym Reacher env
    """

    def __init__(self):
        super().__init__()
        # NOTE: in FetchReach, has_object is False

    def _get_obs(self):
        """
        Overrides _get_obs()

        :return: dict, HER observation
        """
        # positions
        # TODO: add noise?
        achieved_goal = self.get_body_com("fingertip")
        goal = self.get_body_com("target")

        # original obs
        theta = self.sim.data.qpos.flat[:2]

        obs = np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': goal.copy(),
        }


    def compute_reward(self, achieved_goal, goal, info):
        """
        Compute HER reward as per baseline's requirement

        :param achieved_goal: np array, achieved goal
        :param goal: np array, desired goal
        :param info: dict, gym step info (not useful)
        :return: float, reward
        """
        vec = achieved_goal - goal
        reward_dist = - np.linalg.norm(vec, axis=-1)
        # reward_ctrl = - np.square(a).sum()    # TODO: add potential
        return reward_dist
