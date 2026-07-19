import numpy as np
import random
from PIL import Image, ImageDraw
from environment.environment import OvercookedEnvironment
from environment.generator import LevelGenerator
# Import explicit tools from your config
from config import render_tile, render_legend_2col, get_font  

# 1. Base Configuration
TILE_SIZE = 40
TARGET_PLAYABLE_H = 10
TARGET_PLAYABLE_W = 10
H, W = TARGET_PLAYABLE_H + 2, TARGET_PLAYABLE_W + 2
env = OvercookedEnvironment(height=H, width=W)
gen = LevelGenerator(height=H, width=W)

# ── 2. GENERATE AND CONSTRUCT MAP 1 (EASIEST) ─────────────────────────────────
print(f"Initializing OvercookedEnvironment with dimensions: {H}x{W}")
print("Generating easiest layout...")
# easiest_params = [0.0, 0.5, 0.0, 0.5]
easiest_params = [0.0, -1, 0.0, -1]
seed = 42

level_elems_easy = gen.create_level_elements(*easiest_params, seed=seed)
level_des_easy = gen.calculate_element_coords(level_elems_easy)

# Construct a clean grid image using the local TILE_SIZE (bypassing env.render hardcoded sizes)
img_easy = Image.new("RGB", (W * TILE_SIZE, H * TILE_SIZE))
for r in range(H):
    for c in range(W):
        img_easy.paste(render_tile(level_des_easy[r, c], size=TILE_SIZE), (c * TILE_SIZE, r * TILE_SIZE))

easy_structure = [(easiest_params[0], easiest_params[1], easiest_params[2], easiest_params[3]), seed]

# Calculate task cycle metrics for Map 1
env.set_level_template(level_des_easy)
_, info_easy = env.reset()

# ── 2. GENERATE AND CONSTRUCT MAP 2 (HARDEST) ─────────────────────────────────
print("Generating hardest layout...")
hardest_params = [1.0, 1.0, 1.0, 1.0]
# hardest_params = [0, 0, 0, 0]


level_elems_hard = gen.create_level_elements(*hardest_params, seed=seed)
level_des_hard = gen.calculate_element_coords(level_elems_hard)

# Construct a clean grid image for Map 2
img_hard = Image.new("RGB", (W * TILE_SIZE, H * TILE_SIZE))
for r in range(H):
    for c in range(W):
        img_hard.paste(render_tile(level_des_hard[r, c], size=TILE_SIZE), (c * TILE_SIZE, r * TILE_SIZE))

hard_structure = [(hardest_params[0], hardest_params[1], hardest_params[2], hardest_params[3]), seed]

# Calculate task cycle metrics for Map 2
env.set_level_template(level_des_hard)
_, info_hard = env.reset()

# ── 3. GENERATE AND CONSTRUCT MAP 3 (CUSTOM) ─────────────────────────────────
print("Generating custom layout...")
custom_params = [0.5, 0.2, 0.4, -0.3]

level_elems_custom = gen.create_level_elements(*custom_params, seed=seed)
level_des_custom = gen.calculate_element_coords(level_elems_custom)

# Construct a clean grid image for Map 3
img_custom = Image.new("RGB", (W * TILE_SIZE, H * TILE_SIZE))
for r in range(H):
    for c in range(W):
        img_custom.paste(render_tile(level_des_custom[r, c], size=TILE_SIZE), (c * TILE_SIZE, r * TILE_SIZE))

custom_structure = [(custom_params[0], custom_params[1], custom_params[2], custom_params[3]), seed]

env.set_level_template(level_des_custom)
_, info_custom = env.reset()
# ── 4. DASHBOARD GRID & TEXT LAYOUT CONFIGURATION ─────────────────────────────

# Allocate generous visual boxes to prevent any text overlapping
column_width = 550     # Large safe zone width for strings to breath
header_height = 140    # Vertical header block size
padding_x = 40         # Horizontal space separating left and right views
margin_x = 20          # Left margin bounding the entire layout

# Build a single global unified legend from all active layouts
global_grid = np.array([level_des_easy, level_des_hard, level_des_custom])
legend_img = render_legend_2col(global_grid, tile_size=TILE_SIZE)

# Calculate final canvas dimensions accurately
combined_width = margin_x + (column_width * 3) + (padding_x * 2) + margin_x
combined_height = header_height + max(img_easy.height, img_custom.height, img_hard.height) + legend_img.height + 60

# Initialize the master blank canvas
canvas = Image.new("RGB", (combined_width, combined_height), (245, 245, 245))
draw = ImageDraw.Draw(canvas)

# Load custom system typography weights from config.py
font_title = get_font(18)
font_text = get_font(12)
font_code = get_font(11)

# ── 5. DRAW DATA & ASSETS MONTAGES ────────────────────────────────────────────

# --- Left Layout Panel: Map 1 ---
col1_x = margin_x
draw.text((col1_x, 15), "MAP 1: EASIEST LAYOUT", fill=(40, 160, 40), font=font_title)
draw.text((col1_x, 42), f"Parameters: {easy_structure}", fill=(60, 60, 60), font=font_code)

# Breaking down the dictionary into clean readable sub-rows
draw.text((col1_x, 60), f"Obstacles: {level_elems_easy['num_obstacles']} (skew: {level_elems_easy['obs_skew_x']:.2f})", fill=(80, 80, 80), font=font_text)
draw.text((col1_x, 76), f"Extra Resources: {level_elems_easy['num_extra_resources']} (skew: {level_elems_easy['res_skew_x']:.2f})", fill=(80, 80, 80), font=font_text)
draw.text((col1_x, 92), f"Grid Dimensions: {level_elems_easy['width']}x{level_elems_easy['height']}", fill=(80, 80, 80), font=font_text)
draw.text((col1_x, 110), f"Task Cycle Length: {info_easy['cycle_length']} steps", fill=(30, 30, 30), font=font_code)

canvas.paste(img_easy, (col1_x, header_height))


# --- Right Layout Panel: Map 2 ---
col2_x = margin_x + column_width + padding_x
draw.text((col2_x, 15), "MAP 2: HARDEST LAYOUT", fill=(210, 60, 60), font=font_title)
draw.text((col2_x, 42), f"Parameters: {hard_structure}", fill=(60, 60, 60), font=font_code)

# Breaking down the dictionary for the second column
draw.text((col2_x, 60), f"Obstacles: {level_elems_hard['num_obstacles']} (skew: {level_elems_hard['obs_skew_x']:.2f})", fill=(80, 80, 80), font=font_text)
draw.text((col2_x, 76), f"Extra Resources: {level_elems_hard['num_extra_resources']} (skew: {level_elems_hard['res_skew_x']:.2f})", fill=(80, 80, 80), font=font_text)
draw.text((col2_x, 92), f"Grid Dimensions: {level_elems_hard['width']}x{level_elems_hard['height']}", fill=(80, 80, 80), font=font_text)
draw.text((col2_x, 110), f"Task Cycle Length: {info_hard['cycle_length']} steps", fill=(30, 30, 30), font=font_code)

canvas.paste(img_hard, (col2_x, header_height))

# --- Column 3: Map 3 (Custom Layout) ---
col3_x = margin_x + 2 * column_width + 2 * padding_x
draw.text((col3_x, 15), "MAP 3: CUSTOM LAYOUT", fill=(210, 140, 20), font=font_title) # Custom golden color
draw.text((col3_x, 42), f"Parameters: {custom_structure}", fill=(60, 60, 60), font=font_code)
draw.text((col3_x, 60), f"Obstacles: {level_elems_custom['num_obstacles']} (skew: {level_elems_custom['obs_skew_x']:.2f})", fill=(80, 80, 80), font=font_text)
draw.text((col3_x, 76), f"Extra Resources: {level_elems_custom['num_extra_resources']} (skew: {level_elems_custom['res_skew_x']:.2f})", fill=(80, 80, 80), font=font_text)
draw.text((col3_x, 92), f"Grid Dimensions: {level_elems_custom['width']}x{level_elems_custom['height']}", fill=(80, 80, 80), font=font_text)
draw.text((col3_x, 110), f"Task Cycle Length: {info_custom['cycle_length']} steps", fill=(30, 30, 30), font=font_code)
canvas.paste(img_custom, (col3_x, header_height))

# --- Bottom Block: Global Legend Positioning ---
legend_x = (combined_width - legend_img.width) // 2
legend_y = header_height + max(img_easy.height, img_custom.height, img_hard.height) + 30
canvas.paste(legend_img, (legend_x, legend_y))

# ── 6. EXPORT AND SAVE REAL IMAGE ─────────────────────────────────────────────
output_file = "ued_custom_dashboard.png"
canvas.save(output_file)
print(f"Polished UED dashboard successfully generated and saved to: '{output_file}'")

canvas.show()