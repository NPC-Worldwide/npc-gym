# npc-gym

A drop-in gymnasium replacement for training hybrid LLM+ML agents through games, RL environments, and neuroevolution.

```python
import npc_gym as gym

# Use npc-gym environments
env = gym.make("SlimeVolley-v1")
obs = env.reset()
obs, reward, done, info = env.step([1, 0, 1])  # forward + jump

# Or use any existing gymnasium environment — falls through automatically
env = gym.make("CartPole-v1")
```

```bash
pip install -e .
```

## What's Included

**Everything gymnasium has**, plus multi-agent games, partial information, LLM agents, and evolutionary training.

### Spaces
```python
import npc_gym as gym

gym.Box(-1, 1, shape=(12,))      # Continuous
gym.Discrete(n=5)                 # Discrete
gym.MultiBinary(3)                # Binary vectors
gym.MultiDiscrete([3, 4, 5])      # Multi-dim discrete
gym.Text(max_length=1024)         # Natural language
gym.Dict({"obs": gym.Box(0, 1, shape=(4,)), "text": gym.Text()})
```

### Wrappers
```python
env = gym.make("SlimeVolley-v1")
env = gym.TimeLimit(env, max_episode_steps=3000)
env = gym.ClipReward(env, min_r=-1, max_r=1)
env = gym.FlattenObservation(env)

# Base classes for custom wrappers
class MyWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return normalize(obs)
```

### Registration
```python
# Register custom environments
gym.register("MyEnv-v0", entry_point="my_module:MyEnv", max_steps=1000)
env = gym.make("MyEnv-v0")

# List all available
print(gym.list_envs())
```

## Environments

### SlimeVolley-v1 — Physics-Based RL

Direct port of hardmaru/slimevolleygym with identical physics, coordinate system, and baseline AI. Train agents via NEAT neuroevolution or any RL algorithm.

```python
from npc_gym.envs.slime_volleyball import SlimeVolleyEnv, BaselinePolicy
from npcpy.ft.neat import NEATEvolver, NEATConfig, NEATNetwork
from npcpy.ft.engine import get_engine

# Play against the original 120-param RNN baseline
env = SlimeVolleyEnv()  # default opponent = BaselinePolicy

# Or self-play
env = SlimeVolleyEnv(self_play=True)
obs, reward, done, info = env.step(action_right, action_left)

# Train with NEAT (npcpy) — specify compute backend
evolver = NEATEvolver(
    input_size=12, output_size=3,
    config=NEATConfig(population_size=200),
    engine="numpy",  # or "jax", "mlx", "cuda"
)

def fitness(network):
    obs = env.reset(); done = False; total = 0
    while not done:
        raw = network.activate(obs)
        action = [1 if raw[i] > 0 else 0 for i in range(3)]
        obs, reward, done, info = env.step(action)
        total += reward
    return total

best_genome = evolver.run(fitness, generations=500)
```

### InfoPoker-v1 — Partial Information Decomposition

Text chunked into "cards" dealt to players. Players form hypotheses from partial info and bet on confidence.

```python
env = gym.make("InfoPoker-v1",
    source_text="The transformer architecture uses self-attention to process "
                "sequences in parallel. Multi-head attention allows attending "
                "to different representation subspaces...",
    num_players=4,
)
observations, info = env.reset()
```

### More Environments

| Environment | Type | Description |
|------------|------|-------------|
| `SlimeVolley-v1` | Physics/RL | 2D volleyball, NEAT/RL training |
| `InfoPoker-v1` | Multi-agent | Text decomposition poker |
| `HypothesisBJ-v1` | Multi-agent | Hypothesis blackjack |
| `Synthesis-v1` | Multi-agent | Debate tournament with synthesis |
| `GridWorld-v1` | Navigation | Spatial nav with partial observability |
| `Maze-v1` | Navigation | Limited-visibility maze |
| `TicTacToe-v1` | Competitive | Classic board game |
| `ConnectFour-v1` | Competitive | 7x6 board game |
| `Pokemon-v1` | Emulator | Pokemon via PyBoy with vision |

## Training

### NEAT Neuroevolution (via npcpy)

```python
from npcpy.ft.neat import NEATEvolver, NEATConfig

# Multi-backend: numpy, jax, mlx, cuda
evolver = NEATEvolver(
    input_size=12, output_size=3,
    config=NEATConfig(
        population_size=200,
        add_node_rate=0.05,
        add_connection_rate=0.08,
        species_threshold=3.0,
    ),
    engine="mlx",  # Apple Silicon acceleration
)

best = evolver.run(fitness_fn, generations=500)
```

### Genetic Model Evolution

Evolve ensembles of specialized LLM models through gameplay:

```python
from npc_gym.training import TrainingLoop, TrainingConfig

config = TrainingConfig(
    env_class=InfoPoker,
    env_kwargs={"source_text": corpus},
    agent_class=HybridAgent,
    num_agents=4,
    num_epochs=10,
    games_per_epoch=100,
)
loop = TrainingLoop(config)
loop.run()
```

### Trace Collection for DPO

Games produce traces that convert to preference pairs for DPO fine-tuning:

```python
trace = env.get_trace()
pairs = trace.to_preference_pairs(min_reward_gap=0.2)
# [{"prompt": ..., "chosen": ..., "rejected": ..., "reward_gap": ...}]
```

## Architecture

```
npc_gym/
├── core/
│   ├── env.py           # Base Environment (gymnasium-compatible)
│   ├── spaces.py        # Box, Discrete, MultiBinary, MultiDiscrete, Text, Card, Deck
│   ├── compat.py        # Gymnasium compatibility layer (make, register, wrappers)
│   ├── info.py          # Information structures (PID)
│   └── agent.py         # Agent classes (Random, LLM, Hybrid, NPC)
├── envs/
│   ├── slime_volleyball.py  # SlimeVolley with original physics + baseline AI
│   ├── card_game.py         # Base card game
│   ├── info_poker.py        # InfoPoker
│   ├── hypothesis_bj.py     # HypothesisBlackjack
│   ├── synthesis.py         # SynthesisTournament
│   ├── grid_world.py        # GridWorld, Maze, ItemCollector
│   ├── tictactoe.py         # TicTacToe, ConnectFour
│   └── emulator/            # Game emulator environments
├── training/
│   ├── loop.py          # Training orchestrator
│   ├── traces.py        # Trace collection
│   └── evolution.py     # Genetic model evolution
├── streaming/           # Real-time text processing
├── analytics/           # Metrics and visualization
└── rendering/           # Web visualization server
```

## Integration with npcpy

npc-gym builds on [npcpy](https://github.com/cagostino/npcpy):

- **NEAT neuroevolution** (`npcpy.ft.neat`) — evolve neural network topologies
- **Compute engines** (`npcpy.ft.engine`) — numpy, JAX, MLX, CUDA backends
- **LLM interactions** (`npcpy.llm_funcs`) — multi-provider LLM calls
- **NPCArray mixtures** (`npcpy.npc_array`) — ensemble inference, voting, consensus
- **Fine-tuning** (`npcpy.ft`) — SFT, DPO, diffusion, genetic algorithms
- **NPC agents** (`npcpy.npc_compiler`) — agent personas with tools and memory

## License

Apache 2.0
