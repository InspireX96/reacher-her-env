# reacher-her-env

## Usage

Clone this repo and install:

```bash
pip install -e . 
```

Make this env by:

```python
import gym

...

env = gym.make('reacher_her_env:ReacherHerEnv-v2')
```

Train an agent using **OpenAI baselines** HER implementation and show the result:

```bash
python -m baselines.run --alg=her --env=reacher_her_env:ReacherHerEnv-v2 --num_timesteps=100000 --play
```
