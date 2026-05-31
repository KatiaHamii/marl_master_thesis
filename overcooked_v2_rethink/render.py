"""
render.py — Visualise trained IPPO agents in OvercookedV2
==========================================================
Loads saved params from a training results directory, runs one episode,
shows it live in a window, and saves an annotated GIF + first-frame PNG.

Run from MARL/ root:
    JAX_PLATFORMS=cpu uv run --project overcooked_v2_rethink \\
        python overcooked_v2_rethink/render.py \\
        --results overcooked_v2_rethink/results/ippo/cramped_room_v2_full_shaped/2026-05-29_09-52-36 \\
        --layout cramped_room_v2
        
    JAX_PLATFORMS=cpu uv run --project overcooked_v2_rethink \
    python overcooked_v2_rethink/render.py \
    --results overcooked_v2_rethink/results/ippo/asymm_advantages_recipes_center_full_shaped/2026-05-29_16-31-08 \
    --layout asymm_advantages_recipes_center \
    --obs-every 100

    # with a view-size
    JAX_PLATFORMS=cpu uv run --project overcooked_v2_rethink \
    python overcooked_v2_rethink/render.py \
    --results overcooked_v2_rethink/results/ippo/asymm_advantages_recipes_center_view2_shaped/2026-05-30_11-27-11 \
    --layout asymm_advantages_recipes_center \
    --view-size 2 
    
    # stochastic policy (sample actions instead of argmax)
    ... --stochastic

    # no window, just save files
    ... --no-window
"""

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

from overcooked_v2_rethink import OvercookedV2
from overcooked_v2_rethink.networks import ActorCritic
from overcooked_v2_rethink.viz.overcooked_v2_visualizer import OvercookedV2Visualizer

# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_params(results_dir: Path):
    with open(results_dir / "params_agent0.pkl", "rb") as f:
        p0 = pickle.load(f)
    with open(results_dir / "params_agent1.pkl", "rb") as f:
        p1 = pickle.load(f)
    return p0, p1


def _stack_states(states):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)


def _annotate(
    frame: np.ndarray, step: int, total_steps: int, deliveries: int
) -> np.ndarray:
    """Add a header bar above the frame with step / delivery info."""
    H, W, C = frame.shape
    BAR_H = 36

    # Dark header bar
    bar = np.full((BAR_H, W, C), 30, dtype=np.uint8)
    img_bar = Image.fromarray(bar)
    draw = ImageDraw.Draw(img_bar)

    # Pick a font size that fits the bar width
    font = None
    for font_path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            font = ImageFont.truetype(font_path, 18)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    left = f"Step: {step} / {total_steps}"
    right = f"Deliveries: {deliveries}"
    draw.text((10, 12), left, fill=(220, 220, 220), font=font)
    # right-align the deliveries counter
    bbox = draw.textbbox((0, 0), right, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text((W - text_w - 10, 12), right, fill=(100, 220, 100), font=font)

    return np.vstack([np.array(img_bar), frame])


# ── Obs snapshot ──────────────────────────────────────────────────────────────


def _plot_agent_channels(fig, gs_row, obs_np, agent_label, cmaps, ch_titles):
    """Fill one row of a GridSpec with 7 channel heatmaps for one agent."""
    for i in range(7):
        ax  = fig.add_subplot(gs_row[i // 4, i % 4])
        ch  = obs_np[:, :, i]
        vmin, vmax = ch.min(), ch.max()
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5
        im  = ax.imshow(ch, cmap=cmaps[i], vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"[{agent_label}] {ch_titles[i]}", fontsize=7, pad=3)
        for r in range(ch.shape[0]):
            for c in range(ch.shape[1]):
                v    = ch[r, c]
                txt  = str(int(v)) if v == int(v) else f"{v:.0f}"
                norm = (v - vmin) / (vmax - vmin + 1e-9)
                ax.text(c, r, txt, ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white" if norm > 0.55 else "black")
        ax.set_xticks(range(obs_np.shape[1]))
        ax.set_xticklabels([f"c{j}" for j in range(obs_np.shape[1])], fontsize=6)
        ax.set_yticks(range(obs_np.shape[0]))
        ax.set_yticklabels([f"r{j}" for j in range(obs_np.shape[0])], fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)


def _save_obs_snapshot(
    obs_np_0: np.ndarray, obs_np_1: np.ndarray,
    frame: np.ndarray, step: int, deliveries: int, out_path: Path
):
    """Save both agents' 7-channel obs matrices + shared game frame."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpecFromSubplotSpec, GridSpec

    ch_titles = [
        "ch0  self dir", "ch1  self inv",
        "ch2  other dir", "ch3  other inv",
        "ch4  static", "ch5  dynamic", "ch6  extra",
    ]
    cmaps = ["Blues", "Oranges", "Greens", "Purples", "YlOrBr", "Reds", "plasma"]

    fig = plt.figure(figsize=(22, 13))

    # Outer grid: 2 rows (one per agent) on left, game state on right
    outer = GridSpec(2, 2, figure=fig,
                     left=0.03, right=0.99, top=0.91, bottom=0.04,
                     hspace=0.55, wspace=0.08,
                     width_ratios=[4, 1])

    # Agent 0 — top row, left
    gs0 = GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[0, 0],
                                  hspace=0.55, wspace=0.4)
    _plot_agent_channels(fig, gs0, obs_np_0, "agent_0", cmaps, ch_titles)

    # Agent 1 — bottom row, left
    gs1 = GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[1, 0],
                                  hspace=0.55, wspace=0.4)
    _plot_agent_channels(fig, gs1, obs_np_1, "agent_1", cmaps, ch_titles)

    # Game state — right column, spanning both rows
    ax_game = fig.add_subplot(outer[:, 1])
    ax_game.imshow(frame)
    ax_game.set_title("Game state", fontsize=10)
    ax_game.axis("off")

    fig.suptitle(
        f"Both agents obs — step {step}   deliveries: {deliveries}",
        fontsize=12, y=0.97
    )
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ── Episode runner ─────────────────────────────────────────────────────────────


def run_episode(env, params_0, params_1, network, key, greedy: bool = True):
    """Roll out one episode. Returns (states, obs_list, step_deliveries, ep_info)."""
    key, k_reset = jax.random.split(key)
    obs, state = env.reset(k_reset)

    states = [state]
    obs_list = [obs]  # agent observations at each step
    step_deliveries = [0]
    total_reward = 0.0
    deliveries = 0

    for _ in range(env.max_steps):
        key, k0, k1, k_step = jax.random.split(key, 4)

        logits0, _ = network.apply(params_0, obs["agent_0"])
        logits1, _ = network.apply(params_1, obs["agent_1"])

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


# ── Live window ────────────────────────────────────────────────────────────────


def _show_live(annotated_frames: list[np.ndarray], fps: int):
    """Display frames one-by-one in a matplotlib window."""
    import matplotlib

    for backend in ("MacOSX", "Qt5Agg", "GTK3Agg", "WXAgg"):
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue
    import matplotlib.pyplot as plt

    delay = 1.0 / fps
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.ion()
    plt.show()

    im = ax.imshow(annotated_frames[0])
    ax.axis("off")
    fig.tight_layout()

    for frame in annotated_frames:
        im.set_data(frame)
        plt.draw()
        plt.pause(delay)

    plt.ioff()
    plt.title("Episode finished — close window to exit")
    plt.show(block=True)


# ── Main render function ───────────────────────────────────────────────────────


def render(
    results_dir: Path,
    layout: str,
    greedy: bool = True,
    seed: int = 0,
    fps: int = 4,
    hidden_size: int = 128,
    show_window: bool = True,
    obs_every: int = 0,
    view_size: int = None,
):

    params_0, params_1 = _load_params(results_dir)
    env = OvercookedV2(layout=layout, max_steps=400, agent_view_size=view_size)
    network = ActorCritic(n_actions=env.num_actions, gru_hidden=hidden_size)

    key = jax.random.PRNGKey(seed)
    print("Running episode...")
    states, obs_list, step_deliveries, ep_info = run_episode(
        env, params_0, params_1, network, key, greedy=greedy
    )
    print(
        f"  steps={ep_info['steps']}  deliveries={ep_info['deliveries']}  "
        f"reward={ep_info['reward']:.1f}"
    )

    print("Rendering frames...")
    viz = OvercookedV2Visualizer(tile_size=64)
    state_seq = _stack_states(states)
    # pass view_size so the visualizer dims cells outside each agent's view window
    frame_seq = np.array(
        viz.render_sequence(state_seq, agent_view_size=view_size), dtype=np.uint8
    )

    # Annotate every frame with step index and cumulative deliveries
    total_steps = ep_info["steps"]
    annotated_frames = [
        _annotate(frame, step, total_steps, deliv)
        for step, (frame, deliv) in enumerate(zip(frame_seq, step_deliveries))
    ]

    # Save GIF + PNG
    tag = "greedy" if greedy else "stochastic"
    gif_path = results_dir / f"episode_{tag}_seed{seed}.gif"
    png_path = results_dir / f"frame_start_{tag}_seed{seed}.png"

    imageio.mimsave(
        str(gif_path), annotated_frames, format="GIF", duration=int(1000 / fps), loop=0
    )
    imageio.imwrite(str(png_path), annotated_frames[0])

    print(f"  GIF  → {gif_path}")
    print(f"  PNG  → {png_path}")

    # Obs snapshots every N steps
    if obs_every > 0:
        snap_dir = results_dir / f"obs_snapshots_{tag}_seed{seed}"
        snap_dir.mkdir(exist_ok=True)
        total_steps = ep_info["steps"]
        saved = 0
        for step_i, (obs_i, state_i, deliv_i) in enumerate(
            zip(obs_list, states, step_deliveries)
        ):
            if step_i % obs_every == 0:
                frame_i  = np.array(viz._render_state(state_i, view_size), dtype=np.uint8)
                obs_np_0 = np.array(obs_i["agent_0"])
                obs_np_1 = np.array(obs_i["agent_1"])
                out_path = snap_dir / f"obs_step_{step_i:04d}.png"
                _save_obs_snapshot(obs_np_0, obs_np_1, frame_i, step_i, deliv_i, out_path)
                saved += 1
        print(f"  OBS  → {snap_dir}/  ({saved} snapshots, every {obs_every} steps)")

    # Live window
    if show_window:
        print("Showing window (close it to exit)...")
        _show_live(annotated_frames, fps)


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Render trained OvercookedV2 agents — live window + GIF."
    )
    ap.add_argument(
        "--results",
        required=True,
        help="Timestamped run dir (contains params_agent0/1.pkl)",
    )
    ap.add_argument("--layout", default="cramped_room_v2")
    ap.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions (default: greedy argmax)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--fps", type=int, default=4, help="Playback speed in frames/sec (default 4)"
    )
    ap.add_argument(
        "--hidden",
        type=int,
        default=128,
        help="GRU hidden size used during training (default 128)",
    )
    ap.add_argument(
        "--no-window", action="store_true", help="Skip live window, only save files"
    )
    ap.add_argument(
        "--obs-every",
        type=int,
        default=0,
        help="Save obs channel snapshot every N steps (0 = off)",
    )
    ap.add_argument(
        "--view-size",
        type=int,
        default=None,
        help="Partial obs radius used during training (None = full obs). "
        "Must match the value used when training.",
    )
    args = ap.parse_args()

    results_dir = Path(args.results)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {results_dir}")

    print(f"\nResults  : {results_dir}")
    print(f"Layout   : {args.layout}")
    print(
        f"Obs      : {'partial (view-size=%d, window=%dx%d)' % (args.view_size, 2*args.view_size+1, 2*args.view_size+1) if args.view_size else 'full'}"
    )
    print(f"Policy   : {'stochastic' if args.stochastic else 'greedy'}")
    print(f"FPS      : {args.fps}\n")

    render(
        results_dir=results_dir,
        layout=args.layout,
        greedy=not args.stochastic,
        seed=args.seed,
        fps=args.fps,
        hidden_size=args.hidden,
        show_window=not args.no_window,
        obs_every=args.obs_every,
        view_size=args.view_size,
    )


if __name__ == "__main__":
    main()
