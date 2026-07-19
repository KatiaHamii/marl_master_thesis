import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class GridCodes:
    EMPTY        = 0
    AGENT_0      = 2   
    AGENT_1      = 3   
    GOAL         = 4   
    POT          = 5   
    OBSTACLE     = 8   
    PLATE_PILE   = 9   
    INGREDIENT_0 = 10  
    INGREDIENT_1 = 11  
    INGREDIENT_2 = 12

    MAX_OBS_FRAC = 0.9   
    MAX_RES_FRAC = 0.9   

TILE_PX      = 32
C_BG         = ( 20,  20,  20)
C_GREY       = (130, 130, 130)
C_YELLOW     = (240, 220,  40)
C_DARK_GREEN = ( 40, 130,  40)
C_GREEN      = ( 50, 180,  50)
C_ORANGE     = (210, 130,  30)
C_WHITE      = (255, 255, 255)
C_RED        = (210,  60,  60)
C_BLUE       = ( 60, 100, 210)
C_LEGEND_BG  = (245, 245, 245)

_PILE_POS = [(0.50, 0.15), (0.28, 0.42), (0.78, 0.38), (0.38, 0.78), (0.72, 0.74)]

def _circles_on_grey(draw: ImageDraw.Draw, positions, color, r_frac=0.14, s=TILE_PX):
    draw.rectangle([(0, 0), (s - 1, s - 1)], fill=C_GREY)
    r = int(r_frac * s)
    for fx, fy in positions:
        cx, cy = int(fx * s), int(fy * s)
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=color)

def _add_grid_lines(draw: ImageDraw.Draw, s: int):
    lw = max(1, s // 32)
    draw.rectangle([(0, 0), (lw - 1, s - 1)], fill=C_GREY)
    draw.rectangle([(0, 0), (s - 1, lw - 1)], fill=C_GREY)

def render_tile(obj: int, size: int = TILE_PX) -> Image.Image:
    img = Image.new("RGB", (size, size), C_BG)
    draw = ImageDraw.Draw(img)
    s = size

    if obj == GridCodes.OBSTACLE:
        lw  = max(3, s // 12)
        pad = max(4, s // 10)
        draw.line([(pad, pad), (s - 1 - pad, s - 1 - pad)], fill=C_GREY, width=lw)
        draw.line([(s - 1 - pad, pad), (pad, s - 1 - pad)], fill=C_GREY, width=lw)
    elif obj == GridCodes.GOAL:
        draw.rectangle([(0, 0), (s - 1, s - 1)], fill=C_GREY)
        pad = max(2, int(0.10 * s))
        draw.rectangle([(pad, pad), (s - 1 - pad, s - 1 - pad)], fill=C_GREEN)
    elif obj == GridCodes.POT:
        C_POT = (25, 25, 25)
        draw.rectangle([(0, 0), (s - 1, s - 1)], fill=C_GREY)
        pad = max(3, int(0.10 * s))
        body_top = int(0.30 * s)
        draw.rectangle([(pad, body_top), (s - 1 - pad, int(0.90 * s))], fill=C_POT)
        hw = max(3, int(0.09 * s))
        cx = s // 2
        draw.rectangle([(cx - hw, int(0.12 * s)), (cx + hw, body_top)], fill=C_POT)
    elif obj == GridCodes.PLATE_PILE:
        _circles_on_grey(draw, [(0.30, 0.30), (0.72, 0.40), (0.40, 0.72)], C_WHITE, r_frac=0.18, s=size)
    elif obj == GridCodes.INGREDIENT_0:
        _circles_on_grey(draw, _PILE_POS, C_YELLOW, s=size)
    elif obj == GridCodes.INGREDIENT_1:
        _circles_on_grey(draw, _PILE_POS, C_DARK_GREEN, s=size)
    elif obj in (GridCodes.AGENT_0, GridCodes.AGENT_1):
        color = C_RED if obj == GridCodes.AGENT_0 else C_BLUE
        pts = [(int(s * 0.50), int(s * 0.10)), (int(s * 0.88), int(s * 0.86)), (int(s * 0.12), int(s * 0.86))]
        draw.polygon(pts, fill=color)

    _add_grid_lines(draw, s)
    return img

_LABEL = {
    GridCodes.OBSTACLE:     "Wall",
    GridCodes.GOAL:         "Delivery\nStation",
    GridCodes.POT:          "Pot",
    GridCodes.PLATE_PILE:   "Plate",
    GridCodes.INGREDIENT_0: "Ingredient 0",
    GridCodes.INGREDIENT_1: "Ingredient 1",
    GridCodes.AGENT_0:      "Agent 1",
    GridCodes.AGENT_1:      "Agent 2",
}

def get_font(size: int = 13) -> ImageFont.ImageFont:
    for fp in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try: return ImageFont.truetype(fp, size)
        except OSError: continue
    return ImageFont.load_default()

def render_legend_2col(grid: np.ndarray, tile_size: int = TILE_PX) -> Image.Image:
    present = sorted(set(grid.flatten()))
    items   = [(o, _LABEL[o]) for o in present if o in _LABEL]
    font    = get_font(13)
    ncols   = 2
    nrows   = math.ceil(len(items) / ncols)
    pad, label_w = 6, 90
    item_w, item_h  = tile_size + label_w + pad, tile_size + pad
    legend  = Image.new("RGB", (item_w * ncols + pad, item_h * nrows + pad), C_LEGEND_BG)
    draw    = ImageDraw.Draw(legend)
    for i, (obj, label) in enumerate(items):
        col, row = i % ncols, i // ncols
        x, y = pad + col * item_w, pad + row * item_h
        legend.paste(render_tile(obj, size=tile_size), (x, y))
        bbox = draw.textbbox((0, 0), label, font=font)
        th = bbox[3] - bbox[1]
        draw.text((x + tile_size + 5, y + (tile_size - th) // 2), label, fill=(30, 30, 30), font=font)
    return legend

def render_full_map(grid: np.ndarray, tile_size: int = TILE_PX) -> Image.Image:
    H, W = grid.shape
    map_w, map_h = W * tile_size, H * tile_size
    map_img = Image.new("RGB", (map_w, map_h))
    for r in range(H):
        for c in range(W):
            map_img.paste(render_tile(grid[r, c], size=tile_size), (c * tile_size, r * tile_size))
    legend_img = render_legend_2col(grid, tile_size=tile_size)
    final_img = Image.new("RGB", (max(map_w, legend_img.width), map_h + legend_img.height + 10), C_LEGEND_BG)
    final_img.paste(map_img, (0, 0))
    final_img.paste(legend_img, (0, map_h + 10))
    return final_img