# Lightning RND

Reimplementation of ["Exploration by Random Network Distillation"](https://arxiv.org/abs/1810.12894) aiming to train as fast as possible.

# Usage

For a machine with GPUs - install all dependencies from either yml or txt file.

Adjust config yaml file as you wish.

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

Training with only intrinsic reward

#ADDHERE

Training with both intrinsic and extrinsic reward

#ADDHERE

# ToDo

- [x] Separate actor and learner
- [x] Log number of rooms visited
- [x] Add optional V-trace targets correction
- [ ] Add TPU support
- [ ] Add fp16 support 
