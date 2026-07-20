"""
eval_layouts.py — hand-designed benchmark layouts for EVALUATION only.

The canonical Overcooked / OvercookedV2 human layouts (cramped_room,
asymm_advantages, coord_ring, forced_coord, counter_circuit, plus variants).
NOT used for training — the SFL / sfl_jax curriculum generates its own levels.
Kept for evaluating a trained policy zero-shot on fixed human-designed layouts
(the standard UED eval protocol).

    from evaluation.eval_layouts import overcooked_v2_layouts
    layout = overcooked_v2_layouts['cramped_room']
"""

from environment.layouts import Layout


cramped_room = """
WWPWW
OA AO
W   W
WBWXW
"""
asymm_advantages = """
WWWWWWWWW
O WXWOW X
W   P   W
W A PA  W
WWWBWBWWW
"""
coord_ring = """
WWWPW
W A P
BAW W
O   W
WOXWW
"""
forced_coord = """
WWWPW
O WAP
OAW W
B W W
WWWXW
"""
counter_circuit = """
WWWPPWWW
W A    W
B WWWW X
W     AW
WWWOOWWW
"""


# Adapted layouts
cramped_room_v2 = """
WWPWW
0A A1
W   R
WBWXW
"""
asymm_advantages_recipes_center = """
WWWWWWWWW
0 WXR01 X
1   P   W
W A PA  W
WWWBWBWWW
"""
asymm_advantages_recipes_right = """
WWWWWWWWW
0 WXW01 X
1   P   R
W A PA  W
WWWBWBWWW
"""
asymm_advantages_recipes_left = """
WWWWWWWWW
0 WXW01 X
1   P   R
R A PA  W
WWWBWBWWW
"""
two_rooms = """
WWWWWB10W
W   W   R
P A W A W
W   W   X
WWWWWWWWW
"""

# Progressive curriculum: simple 7x7 starting layout (compact, easy to learn)
simple_7x7 = """
WWWWWWW
WPA OAW
WPOBX RW
W      W
W      W
W      W
WWWWWWW
"""



# Other Layouts

long_room = """
WWWWWWWWWWWWWWW
B            AP
0             X
WWWWWWWWWWWWWWW
"""

fun_coordination = """
WWWWWWWWW
0   X   2
RA  P  AW
1   B   3
WWWWWWWWW
"""
more_fun_coordination = """
WWWWWWWWW
W   X   W
RA  P  A1
0   P   2
W   B   W
WWWWWWWWW
"""
fun_symmetries_plates = """
WWWWWWW
B  W  0
R APA X
B  W  1
WWWWWWW
"""
fun_symmetries = """
WWWWBWW
2  W  0
R APA X
2  W  1
WWWWBWW
"""
fun_symmetries1 = """
WWWWWBWW
2  WW  0
R AWPA X
2  WW  1
WWWWWBWW
"""
overcookedv2_demo = """
WWPWW
0A A1
L   R
WBWXW
"""


# Extended Cat-Dog Problem Layouts
grounded_coord_simple = """
WW2WWWWW
W  WB  0
R ALPA X
W  WB  1
WW2WWWWW
"""
grounded_coord_ring = """
WWW2R2WWW
W       W
W WWLWW W
2 0   B 2
RAXAP X R
2 1   B 2
W WWLWW W
W       W
WWW2R2WWW
"""


# Test-Time Protocol Formation Layouts
test_time_simple = """
WW2WWWWW
W  WB  0
R AWPA X
W  WB  1
WW2WWWWW
"""
test_time_wide = """
WWXBWW
0 A  0
1    1
WPWPWW
3 A  3
W    W
WWRWWW
"""


# Demo Cook Layouts
demo_cook_simple = """
WWWWWR2W0WW
0      W  B
W     APA X
1      W  B
WWWWWR2W1WW
"""
demo_cook_wide = """
WWWWBXBWWWW
WWW0 A 1WWW
WWWWWPWWWWW
W    A    W
0  W3R3W  0
W1WWWWWWW1W
"""


overcooked_v2_layouts = {
    # Overcooked-AI layouts
    "cramped_room": Layout.from_string(
        cramped_room, possible_recipes=[[0, 0, 0]], swap_agents=True
    ),
    "asymm_advantages": Layout.from_string(
        asymm_advantages, possible_recipes=[[0, 0, 0]]
    ),
    "coord_ring": Layout.from_string(coord_ring, possible_recipes=[[0, 0, 0]]),
    "forced_coord": Layout.from_string(forced_coord, possible_recipes=[[0, 0, 0]]),
    "counter_circuit": Layout.from_string(
        counter_circuit, possible_recipes=[[0, 0, 0]], swap_agents=True
    ),
    # Adapted layouts
    "cramped_room_v2": Layout.from_string(cramped_room_v2),
    "asymm_advantages_recipes_center": Layout.from_string(
        asymm_advantages_recipes_center
    ),
    "asymm_advantages_recipes_right": Layout.from_string(
        asymm_advantages_recipes_right
    ),
    "asymm_advantages_recipes_left": Layout.from_string(asymm_advantages_recipes_left),
    "two_rooms": Layout.from_string(two_rooms),
    # Other layouts
    "long_room": Layout.from_string(long_room, possible_recipes=[[0, 0, 0]]),
    "fun_coordination": Layout.from_string(
        fun_coordination, possible_recipes=[[0, 0, 2], [1, 1, 3]]
    ),
    "more_fun_coordination": Layout.from_string(
        more_fun_coordination, possible_recipes=[[0, 1, 1], [0, 2, 2]]
    ),
    "fun_symmetries": Layout.from_string(
        fun_symmetries, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    ),
    "fun_symmetries_plates": Layout.from_string(
        fun_symmetries_plates, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    ),
    "fun_symmetries1": Layout.from_string(
        fun_symmetries1, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    ),
    # Extended Cat-Dog Problem Layouts
    "grounded_coord_simple": Layout.from_string(
        grounded_coord_simple, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    ),
    "grounded_coord_ring": Layout.from_string(
        grounded_coord_ring, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    ),
    # Test-Time Protocol Formation Layouts
    "test_time_simple": Layout.from_string(
        test_time_simple, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    ),
    "test_time_wide": Layout.from_string(
        test_time_wide, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    ),
    # Demo Cook Layouts
    "demo_cook_simple": Layout.from_string(
        demo_cook_simple, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    ),
    "demo_cook_wide": Layout.from_string(
        demo_cook_wide, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    ),
}
