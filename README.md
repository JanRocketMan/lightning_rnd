# Lightning RND

Reimplementation of ["Exploration by Random Network Distillation"](https://arxiv.org/abs/1810.12894) aiming to train as fast as possible.

A final project for the course ["Advanced Topics in Deep Reinforcement learning"](http://deeppavlov.ai/rl_course_2020) (a report is available in Russian).

# Usage

Install all dependencies from either yml or txt file.

Adjust config.yml file as you wish (note the "SavePath", "OptimDevice" and "RunDevice" arguments).

Run model training via

```bash
python montezuma_train.py
```

the trained model can be evaluated with

```bash
python montezuma_eval.py
```

# Examples

Montezuma Revenge

Training with both intrinsic and extrinsic rewards

![ext_intr](./videos/usual_training_2.gif)

Training with intrinsic-only reward

![intr](./videos/only_intrinsic_fast.gif)

# ToDo

- [x] Separate actor and learner
- [x] Log number of rooms visited
- [x] Add optional V-trace targets correction
- [ ] Add TPU support
- [ ] Add fp16 support
