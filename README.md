# stepledger

Find silent faults at the environment-trainer boundary in reinforcement learning.

When an environment runs outside the trainer (in another process or on another machine, behind a vector env or a network bridge), a value can change on its way across: an observation rounded to float16, a stale reward, a time limit sent as a termination, a skipped step whose result is filled in. Nothing raises an error, and the policy usually still learns, so training curves do not show it. stepledger keeps a record on each side of the boundary, compares the two step by step, and tells you the env, episode, step and field of the first difference, without needing a fault-free reference run.

```
env 1, episode 2, step 12: reward differs (env gave -11.843963, trainer received -14.121007)
env 3, episode 3, step 7: obs differs (env gave [0.7979722, -0.6026943, -2.675495], trainer received [0.79785156, -0.60253906, -2.6757812])
env 4, episode 0, step 25: terminated, truncated differ (terminated: env gave False, trainer received True; truncated: env gave True, trainer received False)
3 of 8 envs differ
```

## Install

From a checkout of this repository:

```
pip install .            # numpy and gymnasium only
pip install ".[sb3]"     # adds stable-baselines3
```

Python 3.10 or newer, Gymnasium 1.0 or newer. The package imports as `stepledger`; its distribution name is `plugrl-stepledger`. It is not on PyPI yet, and the PyPI project called `stepledger` is an unrelated package.

## Quickstart: Stable-Baselines3

```python
import functools
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stepledger import LedgerEnv
from stepledger.sb3 import LedgerVecEnv

def make_env(i):
    return LedgerEnv(gym.make("Pendulum-v1"), "ledger", index=i)  # env side: outermost, in the worker

if __name__ == "__main__":
    venv = SubprocVecEnv([functools.partial(make_env, i) for i in range(4)])
    venv = LedgerVecEnv(venv, "ledger/trainer.npz")  # trainer side: innermost
    PPO("MlpPolicy", venv).learn(10_000)
    venv.close()  # both sides write their ledgers
```

Then:

```
stepledger compare ledger ledger/trainer.npz
```

## Quickstart: Gymnasium

```python
import functools
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from stepledger import LedgerEnv, LedgerVectorEnv, compare

def make_env(i):
    return LedgerEnv(gym.make("CartPole-v1"), "ledger", index=i)  # env side: outermost, in the worker

if __name__ == "__main__":
    envs = AsyncVectorEnv([functools.partial(make_env, i) for i in range(4)])
    envs = LedgerVectorEnv(envs, "ledger/trainer.npz")  # trainer side: innermost
    envs.reset(seed=0)
    for _ in range(1000):
        envs.step(envs.action_space.sample())
    envs.close()  # both sides write their ledgers
    print(compare("ledger", "ledger/trainer.npz"))
```

Longer versions are in [`examples/`](examples): `gymnasium_async.py` (run it with `--fault` to see a report) and `sb3_pendulum.py`.

## Where the wrappers go

The comparison checks exactly what lies between the two wrappers, so their places decide what is checked.

- `LedgerEnv(env, directory, index)` goes **outermost in the environment's own process**: after your TimeLimit, Monitor and other env-side wrappers, as the last thing before the env is handed to the vector env, the worker or the bridge server. Whatever is inside it counts as the environment; whatever is outside it counts as the boundary. `index` must be the env's position in the trainer's vector env. Each one writes `env-{index}.npz` into `directory` when the env is closed (also on `flush()`, and at interpreter exit if it was never closed).
- `LedgerVectorEnv(vector_env, path)` (Gymnasium) or `stepledger.sb3.LedgerVecEnv(venv, path)` (SB3) goes **innermost on the trainer side**: directly around `AsyncVectorEnv`, `SyncVectorEnv`, `SubprocVecEnv`, `DummyVecEnv` or your bridge client, and inside `VecNormalize`, `VecFrameStack`, `NormalizeObservation` and the like, because those change values on purpose after the boundary. It writes one file with every env's record when closed.
- Both sides must use the same `reward_dtype` (default `np.float32`). `compare` checks that the two sides recorded with the same reward dtype and the same observation and action spaces, and refuses to compare otherwise.

With `SyncVectorEnv` or `DummyVecEnv` both sides live in one process; the boundary is then the vector env itself (its buffers, dtype casts and autoreset logic), which is still worth checking.

## What is recorded and compared

Each side records, per env, a stream of events: `reset(obs)` and `step(action, obs, reward, terminated, truncated)`. In a step event `obs` is what the env returned at that step; at an episode end that is the final observation, not the one from the following reset. End flags are stored as `terminated` and `truncated and not terminated`.

The trainer side follows how the vector env resets envs:

- Gymnasium `NextStep` autoreset (the default): the step after an episode end is a reset; it is recorded as a reset, and the action sent for it, which the env ignores, is not recorded.
- Gymnasium `SameStep` autoreset: on an episode end it records the step with `info["final_obs"][i]`, then the reset.
- Gymnasium `Disabled` autoreset is refused.
- SB3: on `done` it records the step with `infos[i]["terminal_observation"]` and `truncated = infos[i]["TimeLimit.truncated"]`, then the reset. Actions are taken in `step_async`.

Before comparing, values are cast to their declared space: Box, MultiDiscrete and MultiBinary to the space dtype, Discrete to int64, Dict and Tuple leaf by leaf with dict keys sorted. Observations and actions are stored as 64-bit blake2b digests of the cast value (its shape, dtype and bytes). Rewards are cast to `reward_dtype` and stored as their raw bits, which is exact and lets the report print reward values without keeping anything else. Pass `keep_values=True` on both sides to also store observations and actions, so the report prints both sides' values at the first difference.

`compare` reports, per env, the first event where the two streams differ and which fields differ there: `kind` (one side has a reset where the other has a step), `action`, `obs`, `reward`, `terminated`, `truncated`. Episodes are counted from 0 at the first reset; step 0 is the reset and step k the k-th step of the episode, counted on the env side. It also reports streams of different length and envs present on one side only.

`stepledger compare ENV_DIR TRAINER_FILE` (or `python -m stepledger compare ...`) prints the report and exits with 0 when there is no difference, 1 when there is one, and 2 when it cannot compare (missing files, different settings on the two sides).

## What it does not catch

- Faults inside the environment, that is, inside `LedgerEnv`. Both sides see the same wrong value.
- Differences below the precision of the declared dtypes. An observation declared float32 but produced as float64 is compared after the cast to float32. Rewards are compared as float32 by default, because SB3 stores rewards as float32; use `reward_dtype=np.float64` on both sides with trainers that keep float64 rewards.
- Anything not in the event stream: `info` dicts, seeds, reset options, render frames.
- Changes made by wrappers outside the trainer-side ledger. Those are on the trainer side of the comparison by construction.
- Two different values with the same 64-bit digest. The chance per compared field is about 1 in 2^64.

## Cost

Measured with `benchmarks/overhead.py`: a `SyncVectorEnv` of 16 `Pendulum-v1` envs, random actions, 5000 vector steps, median of 5 interleaved runs (i7-1360P laptop, Windows 11, Python 3.12, NumPy 2.5, Gymnasium 1.4).

| ledgers | time per vector step | vs. none |
| --- | --- | --- |
| none | 180 µs | 1.00x |
| `LedgerEnv` on each env | 258 µs | 1.43x |
| `LedgerVectorEnv` | 240 µs | 1.33x |
| both | 325 µs | 1.81x |

That is about 5 µs per env step on the env side and 4 µs on the trainer side. Pendulum takes about 11 µs per env step here, so this is close to the worst case in relative terms; for slower environments the ratio is smaller. Hashing time grows with observation size, at about 0.9 GB/s: about 23 µs per side for an 84x84x3 uint8 frame and 170 µs for 224x224x3.

Storage per event (one reset or one step of one env):

- in memory while recording: 27 bytes (1 kind, three 8-byte digests, 2 flags), plus the values with `keep_values=True`. The trainer side holds every env, so 16 envs x 1M steps is about 430 MB.
- on disk (compressed `.npz`, episode and step counters included): about 20 bytes on Pendulum; about 35 bytes with `keep_values=True`.

## Ledger file format

Each file is an `.npz` without pickles. It holds a JSON `meta` string (format version, side, reward dtype, space descriptions, package version), the env `index` of each stream and stream `offsets`, and these columns, concatenated over streams: `kind` (uint8, 0 reset, 1 step), `h_obs`, `h_act`, `h_rew` (uint64), `terminated`, `truncated` (bool), `episode`, `step` (int64). With `keep_values=True` it also holds `values_obs_{j}` and `values_act_{j}`, one array per space leaf. `stepledger.read(path)` loads one (`.meta`, and `.streams[index].columns`).

## Citation

```bibtex
@inproceedings{gou2026silent,
  title={Detecting Silent Faults at the Environment-Trainer Boundary in Reinforcement Learning},
  author={Gou, Zuo and Lu, Chenhao},
  booktitle={CoRL 2026 Workshop on Everything Beneath the Policy},
  year={2026}
}
```

## Who

stepledger is part of [PlugRL](https://github.com/PlugRL). It was built by [Chenhao Lu](https://github.com/CTP314), [Zuo Gou](https://github.com/tactino) and [Zilin Kang](https://github.com/nothingbutbut).

## License

Apache-2.0
