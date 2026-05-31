# JAX Introduction

## What is JAX?

JAX adds 4 key tools on top of NumPy:

| Tool       | What it does              | Plain English                                      |
| ---------- | ------------------------- | -------------------------------------------------- |
| `jit`      | compiles a function       | "make this 10-100x faster"                         |
| `grad`     | automatic differentiation | "compute gradients for free"                       |
| `vmap`     | automatic batching        | "run this on 64 samples at once without a loop"    |
| `tree_map` | operates on pytrees       | "apply this function to all weight arrays at once" |

---

## Why NOT just use PyTorch?

PyTorch is **stateful** — the network object owns its weights:

```python
net = MyNet()      # weights live inside
net(obs)           # hidden state changes
optimizer.step()   # modifies net in place
```

JAX is **functional** — everything is explicit:

```python
params = init_network(key)         # weights are just a dict
forward(params, obs)               # pass weights in, get output out
new_params = update(params, grads) # return new weights, old ones unchanged
```

No hidden state → JAX can safely compile, parallelise, and differentiate anything.

---

## Pytrees — how JAX stores network weights

A **pytree** is JAX's name for a nested dict/list of arrays.
It is not related to the environment — it stores everything the agent has **learned**.

```python
params = {
    "trunk_w1": jnp.array(...),   # shape (32, 64)
    "trunk_b1": jnp.zeros(64),
    "actor_w":  jnp.array(...),   # shape (64, 4)  → policy π(a|s)
    "critic_w": jnp.array(...),   # shape (64, 1)  → value V(s)
}
```

JAX functions (`grad`, `jit`, `vmap`, `tree_map`) know how to walk through pytrees automatically:

```python
grads      = grad(loss)(params, obs)             # returns dict with same keys
new_params = optax.apply_updates(params, updates) # walks the dict for you
target     = jax.tree_util.tree_map(jnp.copy, params)  # copies every array
```

Each agent has its own separate pytree:
```agents = [
    IPPOAgent(k0, ...),   # agent_0 — has its own params pytree
    IPPOAgent(k1, ...),   # agent_1 — has its own params pytree
]

```

The pytree is the agent's long-term memory — it persists across all episodes. The rollout buffer is short-term memory — cleared after every PPO update
---

## The Actor-Critic network (trunk)

The **trunk** is the shared middle part — both the actor and critic read from it.

```
obs (32,)
    │
    ▼
┌─────────────────────────────┐
│  trunk_w1  (32→64)  + tanh  │   ← learns "what does this state mean?"
│  trunk_w2  (64→64)  + tanh  │   ← deeper representation
└─────────────────────────────┘
         │  64 features
    ┌────┴────┐
    ▼         ▼
actor_w     critic_w
(64→4)      (64→1)
    │             │
 logits        V(s)
π(a|s)      "how good
              is this
              state?"
```

Both heads answer different questions from the **same shared features**:
- **Actor** — which action to take → outputs logits → softmax → probability distribution
- **Critic** — how good is this state → outputs a single number V(s)

Sharing the trunk means they learn the same state representation once, not twice.

---

## Why JAX is useful for MARL specifically

### 1. Multiple agents = multiple networks updating at once

With PyTorch you'd need to carefully manage which `.backward()` belongs to which agent.
In JAX, each agent is just a dict of arrays — no object, no confusion:

```python
# Update agent_0 — completely independent, no interference
new_params_0 = update(params_0, grads_0)

# Update agent_1 — same function, different data
new_params_1 = update(params_1, grads_1)
```

### 2. `vmap` = run all agents in parallel with zero extra code

```python
# Single agent forward pass
actor_critic_forward(params, obs)

# All agents at once — just wrap with vmap, no loop needed
vmap(actor_critic_forward)(all_params, all_obs)
```

### 3. `jit` compiles the entire update step once

```python
@jit
def update_step(params, opt_state, obs, actions, advantages, returns):
    loss, grads = value_and_grad(ppo_loss)(params, ...)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, updates), new_opt_state, loss
```

First call: compiles. Every call after: runs at near-C speed.
Especially valuable in MARL where you run many agents for many episodes.

### 4. Explicit randomness = reproducible multi-agent experiments

```python
master_key = random.PRNGKey(42)
k0, k1 = random.split(master_key)   # each agent gets its own key

agent_0 = IPPOAgent(k0)   # reproducible
agent_1 = IPPOAgent(k1)   # reproducible, different from agent_0
```

In NumPy/PyTorch, random state is global and hard to control across multiple agents.

---

## Key differences at a glance

|                 | NumPy / PyTorch            | JAX                            |
| --------------- | -------------------------- | ------------------------------ |
| Arrays          | mutable (`x[0] = 5`)       | immutable (`x.at[0].set(5)`)   |
| Random state    | global, hidden             | explicit key passed everywhere |
| Network weights | inside an object           | plain dict (pytree)            |
| Gradients       | `loss.backward()`          | `grad(loss)(params)`           |
| Compilation     | optional (`torch.compile`) | `@jit` decorator               |
| Batching        | manual indexing            | `vmap`                         |

---

## One-line summary

> JAX = NumPy syntax + auto-grad + JIT compilation + explicit state —
> making it easy to write fast, reproducible, parallelisable code for
> multiple agents without fighting object-oriented abstractions.
