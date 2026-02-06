# Tailoring PPO to MsPacman

## Background

### MsPacman-v4 Environment

The MsPacman-v4 environment implements the 1982 maze game with the same name, which was by General Computer Corporation (GCC) and published by Midway Manufacturing for arcades [[1](https://en.wikipedia.org/wiki/Ms._Pac-Man),[2](https://www.atariage.com/software_page.php?SoftwareLabelID=320)]. In the game, the player starts in a (fully visible) maze, and has to collect all pellets while avoiding the ghosts. Eating an "energy pill" also allows catching the ghosts which earns points.

(insert image here)

In its gymnasium implementation (default version), it has the following properties:

* Action Space: Discrete(8). Corresponds to the 8 possible meaningful actions in the game (e.g. left, right, ..., noop)
* Observation Space: (210, 160, 3). Corresponds to the RGB image
* Rewards: According to the [manual](https://www.atariage.com/manual_page.php?SystemID=2600&ItemTypeID=&SoftwareLabelID=320&currentPage=6&maxPages=12&currentPage=7)
  * dot: 10 points
  * energy pill: 50 points
  * cherry: 100 points
  * strawberry: 200 points
  * orange: 500 points
  * pretzel: 700 points
  * apple: 1000 points
  * pear: 2000 points
  * banana: 5000 points
  * first caught ghost: 200 points
  * 2nd ghost (same energy pill): 400 points
  * 3rd ghost (same energy pill): 800 points
  * 4th ghost (same energy pill) 1600 points

### PPO

(describe PPO)

## Tailoring experiments

The default hyperparameters are taken from [rl-zoo](../hyperparams/ppo.yml) for atari. They are:

```yml
atari:
  env_wrapper:
    - stable_baselines3.common.atari_wrappers.AtariWrapper
  frame_stack: 4
  policy: 'CnnPolicy'
  n_envs: 8
  n_steps: 128
  n_epochs: 4
  batch_size: 256
  n_timesteps: !!float 1e7
  learning_rate: lin_2.5e-4
  clip_range: lin_0.1
  vf_coef: 0.5
  ent_coef: 0.01
```

They are loaded from [experiments/def_ppo_pac.yml](experiments/def_ppo_pac.yml).

If not otherwise specified, all experiments are run using the [training interface](utils/train_custom.py) with:

* 30 repetitions per variant
* seed 1337
* based on the pre-trained agent for atari (link to agent)

All experiments are run on (hardware specification).
The default version is run as:

```
python utils/train_custom.py --env MsPacmanNoFrameskip-v4 -- conf experiments/def_ppo_pac.yml
```

Complete training commands (including seeds), logs and evaluation results are accessible (here).

### Ram vs RGB observation space

The gymnasium implementation of the Atari games already comes with several [variants](https://gymnasium.farama.org/v0.28.1/environments/atari/ms_pacman/#variants). I was particularly interested in how the observation space affects the performance of the algorithm, because the RAM is a much more compact representation (128 bytes) and might thus be easier/quicker to learn.

(details on observation spaces)

(replicability details)

```
python utils/train_custom.py --env MsPacman-ramNoFrameskip-v4 ...
```

(runtimes)

(training plots)
(evaluation plots)

As expected / surprisingly, during training, the agent using the RAM-observation is seemingly quicker/slower at achieving the first rewards. However, after x episodes, the training performance ...

In contrast, in the evaluation ...

(more discussion)

It therefore seems that with enough training, the rgb image can be interpreted well enough to extract the necessary information. However, even the ram representation potentially contains unnecessary information. Another thing that could be tried is to attempt to transform the rgb or ram info into a matrix-like representation of the gamestate. However, this needs to be fast enough...

### Encouraging longevity

Currently, the player only receives rewards for collected objects. However, I noticed, that at the start of training, the agent dies very quickly. You can see this here:

(training figures. This is a pretend scenario)

In order to encourage evading ghosts, even if no pellets are in the region, I implemented an additional reward of 1 at every timestep. This is implemented in [wrapper/longevity_wrapper.py], with a corresponding config_file [experiments/logevity_ppo_pac.yml].

```
python utils/train_custom.py --env MsPacmanNoFrameskip-v4 --conf experiments/longevity_ppo_pac.yml
```

(runtimes)

(results, discussion, etc)

### Hyperparameter tuning

Since the default parameters are tuned, but on atari games in general, I used the interface to Optuna to tune the hyperparameters specifically for this environment (starting from the pretrained agents and default parameters).

(reproducability, plots)

The new hyperparameters were ...
This might be explained by ...

### Other potential sections/experiments

* If the reward has been changed, does that work better for other types of agents as well?
* Can I train the agent using a curriculum (i.e. easier games first) by using the in-built difficulty setting?
* If I intentionally overfit (train and test on one seed only), what is the maximum performance I can achieve (useful as a baseline).
* What is the performance of a random agent (another useful baseline).
* A known heuristic / I personally think energy pills are undervalued, I completely changed the rewards.
* I removed the noop action, because ...

## Summary and Conclusion

I tried ...
These tailoring approaches were (not) able to produce better performance.
This might be because ... To verify this, we should do the following ...
