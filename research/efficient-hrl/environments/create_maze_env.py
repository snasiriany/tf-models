# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

from environments.ant_maze_env import AntMazeEnv
from environments.point_maze_env import PointMazeEnv

import tensorflow as tf
import gin.tf
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment


@gin.configurable
def create_maze_env(env_name=None, top_down_view=False):
  if env_name in ['Lift', 'Reach']:
    import robosuite as suite
    from robosuite.wrappers.gym_wrapper import GymWrapper
    from robosuite import load_controller_config
    controller_config = load_controller_config(default_controller='OSC_POSITION_YAW')

    controller_config_update=dict(
        position_limits=[
            [-0.30, -0.30, 0.75],
            [0.15, 0.30, 1.15]
        ],
    )
    controller_config.update(controller_config_update)

    obs_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel'] + ['object-state']
    skill_config=dict(
        skills=['ll'],
        aff_penalty_type='add',
        aff_penalty_fac=15.0, # 5.0

        base_config=dict(
            global_xyz_bounds=[
                [-0.30, -0.30, 0.80],
                [0.15, 0.30, 0.95]
            ],
            lift_height=0.95,
            binary_gripper=True,

            aff_threshold=0.06,
            aff_type='dense',
            reach_global=True,
        ),
        ll_config=dict(
            use_ori_params=True,
        ),
    )
    env = GymWrapper(
        suite.make(
            env_name=env_name,  # "Lift" try with other tasks like "Stack" and "Door"
            robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
            has_renderer=False,
            has_offscreen_renderer=False, #False,
            use_camera_obs=False,
            controller_configs=controller_config,
            skill_config=skill_config,

            ignore_done=True,
            reward_shaping=True,
            hard_reset=False,
            control_freq=10,
            camera_heights=512,
            camera_widths=512,
            table_offset=[-0.075, 0, 0.8],
            reward_scale=5.0,
        ),
        keys=obs_keys,
    )
    wrapped_env = gym_wrapper.GymWrapper(env)
    return wrapped_env

  n_bins = 0
  manual_collision = False
  if env_name.startswith('Ego'):
    n_bins = 8
    env_name = env_name[3:]
  if env_name.startswith('Ant'):
    cls = AntMazeEnv
    env_name = env_name[3:]
    maze_size_scaling = 8
  elif env_name.startswith('Point'):
    cls = PointMazeEnv
    manual_collision = True
    env_name = env_name[5:]
    maze_size_scaling = 4
  else:
    assert False, 'unknown env %s' % env_name

  maze_id = None
  observe_blocks = False
  put_spin_near_agent = False
  if env_name == 'Maze':
    maze_id = 'Maze'
  elif env_name == 'Push':
    maze_id = 'Push'
  elif env_name == 'Fall':
    maze_id = 'Fall'
  elif env_name == 'Block':
    maze_id = 'Block'
    put_spin_near_agent = True
    observe_blocks = True
  elif env_name == 'BlockMaze':
    maze_id = 'BlockMaze'
    put_spin_near_agent = True
    observe_blocks = True
  else:
    raise ValueError('Unknown maze environment %s' % env_name)

  gym_mujoco_kwargs = {
      'maze_id': maze_id,
      'n_bins': n_bins,
      'observe_blocks': observe_blocks,
      'put_spin_near_agent': put_spin_near_agent,
      'top_down_view': top_down_view,
      'manual_collision': manual_collision,
      'maze_size_scaling': maze_size_scaling
  }
  gym_env = cls(**gym_mujoco_kwargs)
  gym_env.reset()
  wrapped_env = gym_wrapper.GymWrapper(gym_env)
  return wrapped_env


class TFPyEnvironment(tf_py_environment.TFPyEnvironment):

  def __init__(self, *args, **kwargs):
    super(TFPyEnvironment, self).__init__(*args, **kwargs)

  def start_collect(self):
    pass

  def current_obs(self):
    time_step = self.current_time_step()
    return time_step.observation[0]  # For some reason, there is an extra dim.

  def step(self, actions):
    actions = tf.expand_dims(actions, 0)
    next_step = super(TFPyEnvironment, self).step(actions)
    return next_step.is_last()[0], next_step.reward[0], next_step.discount[0]

  def reset(self):
    return super(TFPyEnvironment, self).reset()
