import os, sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
from pathlib import Path

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from overcooked_v2_rethink import OvercookedV2, Layout, StaticObject
from overcooked_v2_rethink.networks import ActorCritic
from overcooked_v2_rethink.utils import compute_enclosed_spaces
from overcooked_v2_rethink.viz.overcooked_v2_visualizer import OvercookedV2Visualizer

# GridCodes (from overcooked_parametrized_new) → StaticObject int
_GRID_TO_STATIC = {
    0:  int(StaticObject.EMPTY),             # EMPTY
    2:  int(StaticObject.EMPTY),             # AGENT_0 (dynamic)
    3:  int(StaticObject.EMPTY),             # AGENT_1 (dynamic)
    4:  int(StaticObject.GOAL),              # GOAL
    5:  int(StaticObject.POT),               # POT
    8:  int(StaticObject.WALL),              # OBSTACLE
    9:  int(StaticObject.PLATE_PILE),        # PLATE_PILE
    10: int(StaticObject.INGREDIENT_PILE_BASE) + 0,  # INGREDIENT_0
    11: int(StaticObject.INGREDIENT_PILE_BASE) + 1,  # INGREDIENT_1
}

def _grid_to_env(grid: np.ndarray, max_steps: int = 400) -> OvercookedV2:
    """Convert a saved numpy grid (GridCodes) into a ready-to-use OvercookedV2 env."""
    H, W = grid.shape
    static_objects = np.vectorize(lambda c: _GRID_TO_STATIC.get(int(c), 0))(grid).astype(int)

    agent_positions = []
    for code in (2, 3):  # AGENT_0, AGENT_1
        rows, cols = np.where(grid == code)
        if len(rows):
            r, c = int(rows[0]), int(cols[0])
            agent_positions.append((c, r))  # Layout uses (x=col, y=row)
            static_objects[r, c] = int(StaticObject.EMPTY)

    if len(agent_positions) < 2:
        # fallback: place agents at first two empty interior cells
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                if static_objects[r, c] == int(StaticObject.EMPTY) and (c, r) not in agent_positions:
                    agent_positions.append((c, r))
                if len(agent_positions) == 2:
                    break
            if len(agent_positions) == 2:
                break

    layout = Layout(
        agent_positions=agent_positions,
        static_objects=static_objects,
        num_ingredients=2,
        possible_recipes=[[0, 0, 0], [1, 1, 1]],
    )
    return OvercookedV2(layout=layout, max_steps=max_steps)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_params(results_dir: Path):
    with open(results_dir / "params_agent0.pkl", "rb") as f:
        p0 = pickle.load(f)
    with open(results_dir / "params_agent1.pkl", "rb") as f:
        p1 = pickle.load(f)
    return p0, p1


def _stack_states(states):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)


def _annotate(frame: np.ndarray, step: int, total_steps: int, deliveries: int) -> np.ndarray:
    """Додає інформаційну плашку згори над кадром гри."""
    H, W, C = frame.shape
    BAR_H = 36

    bar = np.full((BAR_H, W, C), 30, dtype=np.uint8)
    img_bar = Image.fromarray(bar)
    draw = ImageDraw.Draw(img_bar)

    font = None
    for font_path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            font = ImageFont.truetype(font_path, 16)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    left = f"Step: {step} / {total_steps}"
    right = f"Deliveries: {deliveries}"
    draw.text((10, 10), left, fill=(220, 220, 220), font=font)
    
    bbox = draw.textbbox((0, 0), right, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text((W - text_w - 10, 10), right, fill=(100, 220, 100), font=font)

    return np.vstack([np.array(img_bar), frame])


# ── Спрощений та виправлений Episode Runner ──────────────────────────────────────

def run_episode(env, params_0, params_1, network, key, greedy: bool = True):
    """Генерує один чистий епізод взаємодії агентів з виправленими розмірностями."""
    key, k_reset = jax.random.split(key)
    obs, state = env.reset(k_reset)

    states = [state]
    obs_list = [obs]  
    step_deliveries = [0]
    total_reward = 0.0
    deliveries = 0

    # JIT-компіляція форвард-пасу для швидкості рендерингу
    @jax.jit
    def get_logits(params, single_obs):
        # FIX: Додаємо штучний вимір батчу (1, H, W, 7) щоб уникнути помилки сплощення у ConvEncoder
        batch_obs = single_obs[None, ...]
        logits, _ = network.apply(params, batch_obs)
        return logits[0] # Прибираємо вимір батчу назад після виходу з мережі

    for _ in range(env.max_steps):
        key, k0, k1, k_step = jax.random.split(key, 4)

        logits0 = get_logits(params_0, obs["agent_0"])
        logits1 = get_logits(params_1, obs["agent_1"])

        if greedy:
            a0 = jnp.argmax(logits0)
            a1 = jnp.argmax(logits1)
        else:
            a0 = jax.random.categorical(k0, logits0)
            a1 = jax.random.categorical(k1, logits1)

        obs, state, rewards, dones, _ = env.step_env(
            k_step, state, {"agent_0": a0, "agent_1": a1}
        )
        deliveries += int(state.new_correct_delivery)
        states.append(state)
        obs_list.append(obs)
        step_deliveries.append(deliveries)
        total_reward += float(rewards["agent_0"])

        if dones["__all__"]:
            break

    ep_info = {"steps": len(states), "deliveries": deliveries, "reward": total_reward}
    return states, obs_list, step_deliveries, ep_info


# ── Функція збереження теплокарт 7 базових каналів ───────────────────────────

def _plot_agent_channels(fig, gs_row, obs_np, agent_label, cmaps, ch_titles):
    for i in range(7):
        ax  = fig.add_subplot(gs_row[i // 4, i % 4])
        ch  = obs_np[:, :, i]
        vmin, vmax = ch.min(), ch.max()
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5
        im  = ax.imshow(ch, cmap=cmaps[i], vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"[{agent_label}] {ch_titles[i]}", fontsize=8, pad=3)
        for r in range(ch.shape[0]):
            for c in range(ch.shape[1]):
                v = ch[r, c]
                txt = str(int(v)) if v == int(v) else f"{v:.0f}"
                norm = (v - vmin) / (vmax - vmin + 1e-9)
                ax.text(c, r, txt, ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white" if norm > 0.55 else "black")
        ax.axis("on")
        fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)


def _save_obs_snapshot(obs_np_0: np.ndarray, obs_np_1: np.ndarray, frame: np.ndarray, step: int, deliveries: int, out_path: Path):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpecFromSubplotSpec, GridSpec

    ch_titles = [
        "ch0: self dir", "ch1: self inv",
        "ch2: other dir", "ch3: other inv",
        "ch4: static tile", "ch5: dynamic item", "ch6: extra/timer",
    ]
    cmaps = ["Blues", "Oranges", "Greens", "Purples", "YlOrBr", "Reds", "plasma"]

    fig = plt.figure(figsize=(20, 11))
    outer = GridSpec(2, 2, figure=fig, left=0.02, right=0.98, top=0.92, bottom=0.04, hspace=0.4, wspace=0.1, width_ratios=[3, 1])

    gs0 = GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[0, 0], hspace=0.4, wspace=0.3)
    _plot_agent_channels(fig, gs0, obs_np_0, "Agent 0", cmaps, ch_titles)

    gs1 = GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[1, 0], hspace=0.4, wspace=0.3)
    _plot_agent_channels(fig, gs1, obs_np_1, "Agent 1", cmaps, ch_titles)

    ax_game = fig.add_subplot(outer[:, 1])
    ax_game.imshow(frame)
    ax_game.set_title("Current Game State Render", fontsize=12)
    ax_game.axis("off")

    fig.suptitle(f"Multi-Agent Compact Observations Breakdown — Step {step} (Total Deliveries: {deliveries})", fontsize=14, y=0.98)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# # ── Головний метод рендерингу ──────────────────────────────────────────────────

# def render(results_dir: Path, layout: str, greedy: bool = True, seed: int = 0, fps: int = 4, show_window: bool = True, obs_every: int = 0):
#     params_0, params_1 = _load_params(results_dir)
    
#     # 1. Create the base OvercookedV2 instance
#     env = OvercookedV2(layout=layout, max_steps=400)
    
#     # 2. IMPORT PARAMETRIZED GENERATOR (Same as SFLTrainer)
#     from overcooked_v2_rethink.overcooked_parametrized_current import ParametrizedOvercooked, _LAYOUTS as _raw_layouts
#     from overcooked_v2_rethink.common import StaticObject
#     from overcooked_v2_rethink.utils import compute_enclosed_spaces

#     _G = ParametrizedOvercooked.codes
#     _TO_STATIC = {
#         _G.EMPTY:        int(StaticObject.EMPTY),
#         _G.GOAL:         int(StaticObject.GOAL),
#         _G.POT:          int(StaticObject.POT),
#         _G.OBSTACLE:     int(StaticObject.WALL),
#         _G.PLATE_PILE:   int(StaticObject.PLATE_PILE),
#         _G.INGREDIENT_0: int(StaticObject.INGREDIENT_PILE_BASE) + 0,
#         _G.INGREDIENT_1: int(StaticObject.INGREDIENT_PILE_BASE) + 1,
#         _G.AGENT_0:      int(StaticObject.EMPTY),
#         _G.AGENT_1:      int(StaticObject.EMPTY),
#     }

#     # Get the base string layout representation
#     base_layout_str = _raw_layouts.get(layout)
#     if base_layout_str is None:
#         raise ValueError(f"Layout '{layout}' not found in parametrized configurations.")

#     # Build the generator instance
#     param_env = ParametrizedOvercooked(base_grid=ParametrizedOvercooked.from_string(base_layout_str), seed=seed)

#     # 3. GENERATE A VALID LEVEL CONFIGURATION
#     import jax
#     init_key = jax.random.PRNGKey(seed)
#     grid, _ = param_env._generate_from_key(init_key)

#     # 4. CONVERT GRID TO STATIC OBJECTS & AGENT POSITIONS
#     H, W = grid.shape
#     static_objects = np.vectorize(_TO_STATIC.get)(grid).astype(int)
#     agent_positions = []
#     for code in [ParametrizedOvercooked.codes.AGENT_0, ParametrizedOvercooked.codes.AGENT_1]:
#         rows, cols = np.where(grid == code)
#         if len(rows):
#             agent_positions.append((int(cols[0]), int(rows[0])))

#     # 5. FORCE THE ENVIRONMENT TO USE THIS PARAMETRIZED LAYOUT
#     # Also recompute enclosed_spaces so env internals stay consistent with new static_objects
#     env.layout.static_objects = static_objects
#     env.enclosed_spaces = compute_enclosed_spaces(static_objects == StaticObject.EMPTY)
#     if agent_positions:
#         env.layout.agent_positions = agent_positions

#     # The rest of your code remains unchanged
#     network = ActorCritic(n_actions=env.num_actions, obs_mode="rich")
#     key = jax.random.PRNGKey(seed)
    
#     print(f"[RENDER] Running evaluation episode on SFL-generated layout: '{layout}'...")
#     states, obs_list, step_deliveries, ep_info = run_episode(env, params_0, params_1, network, key, greedy=greedy)
#     print(f"[RESULT] Steps: {ep_info['steps']} | Deliveries: {ep_info['deliveries']} | Shaped Reward: {ep_info['reward']:.2f}")

#     print("[RENDER] Generating visual frames...")
#     viz = OvercookedV2Visualizer(tile_size=64)
#     state_seq = _stack_states(states)
#     frame_seq = np.array(viz.render_sequence(state_seq), dtype=np.uint8)

#     annotated_frames = [
#         _annotate(frame, step, ep_info['steps'], deliv)
#         for step, (frame, deliv) in enumerate(zip(frame_seq, step_deliveries))
#     ]

#     tag = "greedy" if greedy else "stochastic"
#     gif_path = results_dir / f"evaluation_{tag}_seed{seed}.gif"
#     png_path = results_dir / f"first_frame_{tag}_seed{seed}.png"

#     imageio.mimsave(str(gif_path), annotated_frames, format="GIF", duration=int(1000 / fps), loop=0)
#     imageio.imwrite(str(png_path), annotated_frames[0])
#     print(f"[SAVED] Animated GIF: {gif_path}")
#     print(f"[SAVED] Static PNG  : {png_path}")

#     if obs_every > 0:
#         snap_dir = results_dir / f"snapshots_{tag}_seed{seed}"
#         snap_dir.mkdir(exist_ok=True)
#         for step_i, (obs_i, state_i, deliv_i) in enumerate(zip(obs_list, states, step_deliveries)):
#             if step_i % obs_every == 0:
#                 frame_i = np.array(viz._render_state(state_i), dtype=np.uint8)
#                 out_path = snap_dir / f"obs_step_{step_i:04d}.png"
#                 _save_obs_snapshot(np.array(obs_i["agent_0"]), np.array(obs_i["agent_1"]), frame_i, step_i, deliv_i, out_path)
#         print(f"[SAVED] Spatial observation breakdowns in: {snap_dir}/")

#     if show_window:
#         import matplotlib.pyplot as plt
#         print("[RENDER] Opening interactive playback window...")
        
#         fig, ax = plt.subplots(figsize=(6, 6))
#         plt.ion()
#         plt.show()
#         im = ax.imshow(annotated_frames[0])
#         ax.axis("off")
#         fig.tight_layout()

#         for frame in annotated_frames:
#             im.set_data(frame)
#             plt.draw()
#             plt.pause(1.0 / fps)
        
#         plt.ioff()
#         plt.title("Finished! Close window to exit.")
#         plt.show(block=True)

def render(results_dir: Path, layout: str, greedy: bool = True, seed: int = 0, fps: int = 4, show_window: bool = True, obs_every: int = 0):
    params_0, params_1 = _load_params(results_dir)
    
    # 1. Create the clean target environment
    env = OvercookedV2(layout=layout, max_steps=400)
    
    # 2. Import everything from your actual environment layout files
    from overcooked_v2_rethink.overcooked_parametrized_current import ParametrizedOvercooked, _LAYOUTS as _raw_layouts
    from overcooked_v2_rethink.overcooked_v2_current import compute_enclosed_spaces
    from overcooked_v2_rethink.common import StaticObject
    
    # Get raw string representation of the layout
    base_layout_str = _raw_layouts.get(layout)
    if base_layout_str is None:
        raise ValueError(f"Layout '{layout}' not found in parametrized configurations.")
        
    # 3. Instantiate the exact same generator used in SFLTrainer
    # This guarantees num_agents, num_pots, num_plates are correctly extracted from the base layout string
    param_env = ParametrizedOvercooked(base_grid=ParametrizedOvercooked.from_string(base_layout_str), seed=seed)
    
    # 4. Generate the exact same grid and parameters using fold_in logic (Mirroring SFLTrainer)
    # We simulate exactly 1 attempt (_counter=1) to fetch the deterministic first level for this seed
    param_env._counter = 1 
    init_key = jax.random.fold_in(jax.random.PRNGKey(seed), param_env._counter)
    grid, params = param_env._generate_from_key(init_key)
    
    # 5. Convert the generated grid safely using your custom GridCodes to StaticObject integers
    H, W = grid.shape
    safe_mapping = lambda code: _TO_STATIC.get(code, int(StaticObject.EMPTY))
    static_objects = np.vectorize(safe_mapping)(grid).astype(int)
    
    # Extract structural positions for dynamic agents
    agent_positions = []
    for code in [ParametrizedOvercooked.codes.AGENT_0, ParametrizedOvercooked.codes.AGENT_1]:
        rows, cols = np.where(grid == code)
        if len(rows):
            r, c = int(rows[0]), int(cols[0])
            agent_positions.append((c, r)) # Layout expects (x=col, y=row)
            # Crucial: Clear floor under agents so they don't block themselves as static obstacles
            static_objects[r, c] = int(StaticObject.EMPTY)
            
    # 6. Apply everything back into the environment metadata
    env.layout.static_objects = static_objects
    if agent_positions:
        env.layout.agent_positions = agent_positions
        
    # Synchronize enclosed navigation spaces to avoid trace bugs
    env.enclosed_spaces = compute_enclosed_spaces(static_objects == int(StaticObject.EMPTY))

    # ── Standard Evaluation Loop Execution (Unchanged) ────────────────────────
    network = ActorCritic(n_actions=env.num_actions, obs_mode="rich")
    key = jax.random.PRNGKey(seed)
    
    print(f"[RENDER] Running evaluation episode on SFL synchronized layout: '{layout}'...")
    states, obs_list, step_deliveries, ep_info = run_episode(env, params_0, params_1, network, key, greedy=greedy)
    print(f"[RESULT] Steps: {ep_info['steps']} | Deliveries: {ep_info['deliveries']} | Shaped Reward: {ep_info['reward']:.2f}")

    print("[RENDER] Generating visual frames...")
    viz = OvercookedV2Visualizer(tile_size=64)
    state_seq = _stack_states(states)
    frame_seq = np.array(viz.render_sequence(state_seq), dtype=np.uint8)

    annotated_frames = [
        _annotate(frame, step, ep_info['steps'], deliv)
        for step, (frame, deliv) in enumerate(zip(frame_seq, step_deliveries))
    ]

    tag = "greedy" if greedy else "stochastic"
    gif_path = results_dir / f"evaluation_{tag}_seed{seed}.gif"
    png_path = results_dir / f"first_frame_{tag}_seed{seed}.png"

    imageio.mimsave(str(gif_path), annotated_frames, format="GIF", duration=int(1000 / fps), loop=0)
    imageio.imwrite(str(png_path), annotated_frames[0])
    print(f"[SAVED] Animated GIF: {gif_path}")
    print(f"[SAVED] Static PNG  : {png_path}")

    if obs_every > 0:
        snap_dir = results_dir / f"snapshots_{tag}_seed{seed}"
        snap_dir.mkdir(exist_ok=True)
        for step_i, (obs_i, state_i, deliv_i) in enumerate(zip(obs_list, states, step_deliveries)):
            if step_i % obs_every == 0:
                frame_i = np.array(viz._render_state(state_i), dtype=np.uint8)
                out_path = snap_dir / f"obs_step_{step_i:04d}.png"
                _save_obs_snapshot(np.array(obs_i["agent_0"]), np.array(obs_i["agent_1"]), frame_i, step_i, deliv_i, out_path)
        print(f"[SAVED] Spatial observation breakdowns in: {snap_dir}/")

    if show_window:
        import matplotlib.pyplot as plt
        print("[RENDER] Opening interactive playback window...")
        
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.ion()
        plt.show()
        im = ax.imshow(annotated_frames[0])
        ax.axis("off")
        fig.tight_layout()

        for frame in annotated_frames:
            im.set_data(frame)
            plt.draw()
            plt.pause(1.0 / fps)
        
        plt.ioff()
        plt.title("Finished! Close window to exit.")
        plt.show(block=True)


def render_from_pkl(
    results_dir: Path,
    reset_id: int = None,
    index: int = 0,
    greedy: bool = True,
    seed: int = 0,
    fps: int = 5,
    show_window: bool = True,
    max_steps: int = 400,
):
    """Render an episode on a specific env saved in env_params.pkl."""
    pkl_path = results_dir / "env_params.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"No env_params.pkl in {results_dir}")

    with open(pkl_path, "rb") as f:
        log = pickle.load(f)  # list of (vec, reset_id, grid, gen_counter)

    if reset_id is not None:
        matches = [(i, entry) for i, entry in enumerate(log) if entry[1] == reset_id]
        if not matches:
            raise ValueError(f"reset_id={reset_id} not found in pkl ({len(log)} entries)")
        idx, (vec, rid, grid, gc) = matches[0]
        print(f"[PKL] Found reset_id={rid} at pkl index {idx}  gen_counter={gc}")
    else:
        if index >= len(log):
            raise ValueError(f"index={index} out of range (pkl has {len(log)} entries)")
        vec, rid, grid, gc = log[index]
        print(f"[PKL] Using index={index}  reset_id={rid}  gen_counter={gc}")

    print(f"[PKL] Grid shape: {grid.shape}  vec={np.round(vec, 3)}")

    env = _grid_to_env(grid, max_steps=max_steps)
    params_0, params_1 = _load_params(results_dir)
    network = ActorCritic(n_actions=env.num_actions, obs_mode="rich")
    key = jax.random.PRNGKey(seed)

    print("[RENDER] Running episode...")
    states, obs_list, step_deliveries, ep_info = run_episode(env, params_0, params_1, network, key, greedy=greedy)
    print(f"[RESULT] Steps: {ep_info['steps']} | Deliveries: {ep_info['deliveries']} | Reward: {ep_info['reward']:.2f}")

    print("[RENDER] Generating frames...")
    viz = OvercookedV2Visualizer(tile_size=64)
    state_seq = _stack_states(states)
    frame_seq = np.array(viz.render_sequence(state_seq), dtype=np.uint8)

    annotated_frames = [
        _annotate(frame, step, ep_info["steps"], deliv)
        for step, (frame, deliv) in enumerate(zip(frame_seq, step_deliveries))
    ]

    tag = "greedy" if greedy else "stochastic"
    gif_path = results_dir / f"pkl_rid{rid}_{tag}_seed{seed}.gif"
    png_path = results_dir / f"pkl_rid{rid}_{tag}_seed{seed}.png"

    imageio.mimsave(str(gif_path), annotated_frames, format="GIF", duration=int(1000 / fps), loop=0)
    imageio.imwrite(str(png_path), annotated_frames[0])
    print(f"[SAVED] GIF : {gif_path}")
    print(f"[SAVED] PNG : {png_path}")

    if show_window:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.ion()
        plt.show()
        im = ax.imshow(annotated_frames[0])
        ax.axis("off")
        fig.tight_layout()
        for frame in annotated_frames:
            im.set_data(frame)
            plt.draw()
            plt.pause(1.0 / fps)
        plt.ioff()
        plt.title("Finished! Close window to exit.")
        plt.show(block=True)


def main():
    ap = argparse.ArgumentParser(description="Render trained OvercookedV2 IPPO agents.")
    ap.add_argument("--results", required=True, help="Path to results folder containing params_agent0/1.pkl")

    # ── Mode A: fixed named layout ──────────────────────────────────────
    ap.add_argument("--layout", default=None, help="Named layout (plain IPPO runs)")

    # ── Mode B: env from env_params.pkl ────────────────────────────────
    ap.add_argument("--from-pkl", action="store_true", help="Load env from env_params.pkl instead of --layout")
    ap.add_argument("--reset-id", type=int, default=None, help="Select env by reset_id from CSV/pkl")
    ap.add_argument("--index",    type=int, default=0,    help="Select env by position in pkl list (0-indexed)")

    # ── Shared ──────────────────────────────────────────────────────────
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--fps",        type=int, default=5)
    ap.add_argument("--no-window",  action="store_true")
    ap.add_argument("--obs-every",  type=int, default=0)
    ap.add_argument("--max-steps",  type=int, default=400)
    args = ap.parse_args()

    if args.from_pkl:
        render_from_pkl(
            results_dir=Path(args.results),
            reset_id=args.reset_id,
            index=args.index,
            greedy=not args.stochastic,
            seed=args.seed,
            fps=args.fps,
            show_window=not args.no_window,
            max_steps=args.max_steps,
        )
    else:
        if args.layout is None:
            ap.error("--layout is required unless --from-pkl is set")
        render(
            results_dir=Path(args.results),
            layout=args.layout,
            greedy=not args.stochastic,
            seed=args.seed,
            fps=args.fps,
            show_window=not args.no_window,
            obs_every=args.obs_every,
        )

if __name__ == "__main__":
    main()