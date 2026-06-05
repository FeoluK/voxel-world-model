"""
render_isometric3.py — higher-fidelity isometric voxel renderer (PIL/numpy, NO matplotlib).

A visual upgrade over render_isometric2.py:

  (a) COMPLETE per-block-type palette keyed on minecraft:* base names. Every base name
      in global_palette.json resolves to a realistic average colour (explicit table for the
      common/important blocks; a keyword-driven heuristic fills the long tail) so nothing
      falls back to flat grey. Top/side colours differ for grass/podzol/mycelium/logs.
  (b) Plants / flowers / torches render as thin X-shaped billboards or reduced geometry,
      not full solid cubes (short grass, fern, dandelion, poppy, tulips, torch, sapling,
      mushrooms, sugar cane, dead bush, etc.).
  (c) Ambient occlusion (voxels with many solid neighbours / in crevices get darkened) +
      anti-aliasing via 2x supersampling then high-quality downscale.
  (d) Subtle per-face ordered dither + slight leaf transparency so canopies read as foliage
      rather than solid lego bricks.

Conventions (match the existing pipeline):
  * AIR = id 0 (minecraft:void_air / air / cave_air).
  * Apply np.swapaxes(grid, 1, 2) before rendering (Y-up swap) — done here when --swap_yz
    or via the programmatic flag (default True for these grids).
  * id -> name from global_palette.json (dict name->id). Strip 'minecraft:' and '[state]'.

Usage:
    python render_isometric3.py --voxels grids.npz --key grids --frame 0 --output out.png
    python render_isometric3.py --voxels roll.npz --key voxels --frame 0 --output out.png
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------
AIR_NAMES = {"air", "void_air", "cave_air", ""}

DEFAULT_PALETTE_JSON = "/Users/frozone/Documents/IWorld/v7_github_staging/src/global_palette.json"

# Isometric projection — larger tiles than v2 for a higher-res look.
TILE_H = 12          # half-width of the top diamond (px) at render (pre-supersample) scale
TILE_DEPTH = 11      # vertical depth of side faces (px)

# Face shading (directional light from upper-left-front).
SHADE_TOP = 1.00
SHADE_LEFT = 0.80
SHADE_RIGHT = 0.64

SUPERSAMPLE = 2      # render at Nx then downscale (anti-aliasing)
BG_COLOR = (247, 249, 252)

# ---------------------------------------------------------------------------
# Explicit colour table (realistic average Minecraft block colours).
# Keyed on base name. (top_rgb, side_rgb); if side is None it equals top.
# These cover everything common in the data plus a broad spread of block types.
# ---------------------------------------------------------------------------
def _c(top, side=None):
    return (tuple(top), tuple(side) if side is not None else tuple(top))

BLOCK_COLORS: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    # --- terrain / stone family ---
    "stone":            _c((131, 131, 131)),
    "smooth_stone":     _c((158, 158, 158)),
    "cobblestone":      _c((127, 127, 127)),
    "mossy_cobblestone":_c((110, 121, 100)),
    "andesite":         _c((136, 137, 137)),
    "polished_andesite":_c((146, 148, 148)),
    "diorite":          _c((188, 188, 190)),
    "polished_diorite": _c((198, 199, 201)),
    "granite":          _c((153, 109, 90)),
    "polished_granite": _c((163, 119, 100)),
    "gravel":           _c((131, 124, 122)),
    "clay":             _c((159, 165, 178)),
    "dirt":             _c((134, 96, 67)),
    "coarse_dirt":      _c((118, 85, 60)),
    "farmland":         _c((97, 64, 38), (134, 96, 67)),
    "grass_path":       _c((148, 121, 65), (134, 96, 67)),
    "podzol":           _c((90, 64, 28), (109, 78, 47)),
    "mycelium":         _c((121, 105, 116), (110, 90, 95)),
    "netherrack":       _c((104, 50, 47)),
    "basalt":           _c((73, 73, 79)),
    "polished_basalt":  _c((86, 86, 92)),
    "blackstone":       _c((42, 38, 44)),
    "bedrock":          _c((85, 85, 85)),
    "obsidian":         _c((20, 17, 33)),
    "crying_obsidian":  _c((36, 18, 66)),
    "end_stone":        _c((221, 223, 165)),
    "soul_sand":        _c((84, 64, 51)),
    "soul_soil":        _c((92, 70, 56)),
    "magma_block":      _c((142, 73, 47)),
    "terracotta":       _c((152, 94, 67)),

    # --- grass block: green top, dirt sides ---
    "grass_block":      _c((114, 169, 76), (134, 96, 67)),

    # --- sand / sandstone ---
    "sand":             _c((219, 207, 163)),
    "red_sand":         _c((190, 102, 47)),
    "sandstone":        _c((216, 203, 157)),
    "smooth_sandstone": _c((221, 209, 163)),
    "cut_sandstone":    _c((218, 206, 160)),
    "chiseled_sandstone":_c((214, 201, 155)),

    # --- ores ---
    "coal_ore":         _c((116, 116, 116)),
    "iron_ore":         _c((152, 137, 121)),
    "gold_ore":         _c((151, 142, 100)),
    "diamond_ore":      _c((123, 165, 165)),
    "emerald_ore":      _c((117, 151, 120)),
    "lapis_ore":        _c((107, 117, 141)),
    "redstone_ore":     _c((140, 113, 113)),
    "coal_block":       _c((25, 25, 25)),
    "iron_block":       _c((220, 220, 220)),
    "gold_block":       _c((249, 224, 90)),
    "diamond_block":    _c((110, 219, 214)),
    "emerald_block":    _c((66, 200, 110)),
    "lapis_block":      _c((39, 67, 138)),
    "redstone_block":   _c((175, 30, 16)),

    # --- leaves (vivid greens per species) ---
    "oak_leaves":       _c((63, 142, 42)),
    # birch_leaves was (128,167,85): too desaturated -> read grey/olive next to oak.
    "birch_leaves":     _c((118, 178, 70)),
    "spruce_leaves":    _c((57, 102, 64)),
    "dark_oak_leaves":  _c((61, 119, 41)),
    "jungle_leaves":    _c((61, 145, 36)),
    "acacia_leaves":    _c((104, 142, 38)),

    # --- logs (bark side / ring top) ---
    "oak_log":          _c((123, 99, 60), (104, 80, 47)),
    # birch_log was (215,213,205)/(197,197,186): near-achromatic -> renders as grey
    # under side shading (RIGHT*0.64 -> (126,126,119)). Warm it so bark reads cream, not stone.
    "birch_log":        _c((230, 222, 196), (210, 198, 168)),
    "spruce_log":       _c((90, 67, 41), (58, 42, 24)),
    "dark_oak_log":     _c((76, 58, 36), (52, 39, 24)),
    "jungle_log":       _c((116, 91, 53), (85, 65, 36)),
    "acacia_log":       _c((124, 80, 56), (105, 92, 84)),
    "oak_wood":         _c((104, 80, 47)),
    "spruce_wood":      _c((58, 42, 24)),
    "stripped_oak_log": _c((184, 150, 96)),
    "stripped_birch_log":_c((196, 176, 121)),
    "stripped_spruce_log":_c((116, 90, 58)),
    "stripped_dark_oak_log":_c((96, 73, 46)),
    "stripped_jungle_log":_c((172, 134, 92)),
    "stripped_acacia_log":_c((183, 92, 56)),

    # --- planks ---
    "oak_planks":       _c((162, 130, 78)),
    "birch_planks":     _c((196, 175, 121)),
    "spruce_planks":    _c((114, 84, 48)),
    "dark_oak_planks":  _c((66, 43, 20)),
    "jungle_planks":    _c((160, 115, 80)),
    "acacia_planks":    _c((168, 90, 50)),

    # --- water / ice / lava ---
    "water":            _c((58, 110, 206)),
    "ice":              _c((145, 183, 246)),
    "packed_ice":       _c((141, 180, 245)),
    "blue_ice":         _c((116, 168, 244)),
    "frosted_ice":      _c((160, 195, 248)),
    "snow":             _c((243, 248, 252)),
    "snow_block":       _c((243, 248, 252)),
    "lava":             _c((221, 108, 32)),

    # --- common building ---
    "bricks":           _c((150, 97, 83)),
    "stone_bricks":     _c((122, 121, 117)),
    "mossy_stone_bricks":_c((110, 118, 100)),
    "cracked_stone_bricks":_c((118, 117, 112)),
    "chiseled_stone_bricks":_c((120, 119, 114)),
    "quartz_block":     _c((235, 232, 225)),
    "bookshelf":        _c((164, 132, 78), (151, 110, 64)),
    "glass":            _c((197, 226, 232)),
    "glowstone":        _c((194, 154, 88)),
    "sea_lantern":      _c((188, 203, 196)),
    "hay_block":        _c((166, 138, 31), (150, 122, 24)),
    "pumpkin":          _c((201, 121, 24)),
    "carved_pumpkin":   _c((201, 121, 24)),
    "jack_o_lantern":   _c((214, 145, 44)),
    "melon":            _c((110, 140, 38)),
    "nether_bricks":    _c((44, 22, 26)),
    "prismarine":       _c((99, 152, 137)),
    "prismarine_bricks":_c((96, 160, 146)),
    "dark_prismarine":  _c((52, 86, 70)),
    "sponge":           _c((196, 192, 74)),
    "wet_sponge":       _c((158, 168, 70)),
    "cactus":           _c((85, 127, 47)),
    "bone_block":       _c((228, 224, 205)),
    "slime_block":      _c((111, 192, 91)),
    "honey_block":      _c((228, 161, 56)),
    "tnt":              _c((167, 56, 38)),
    "redstone_lamp":    _c((96, 62, 36)),
    "note_block":       _c((104, 70, 47)),
    "crafting_table":   _c((124, 84, 51)),
    "furnace":          _c((111, 111, 111)),
    "chest":            _c((155, 116, 58)),
    "spawner":          _c((37, 49, 61)),
    "nether_wart_block":_c((114, 16, 18)),
    "warped_wart_block":_c((22, 121, 113)),
    "shroomlight":      _c((233, 156, 80)),
    "soul_sand_valley": _c((84, 64, 51)),

    # --- exotic / rare cubes (rare in our data, but avoid grey fallback) ---
    "crimson_nylium":   _c((130, 38, 48), (104, 50, 47)),
    "warped_nylium":    _c((43, 119, 110), (104, 50, 47)),
    "honeycomb_block":  _c((229, 152, 47)),
    "dried_kelp_block": _c((52, 58, 44)),
    "polished_blackstone":_c((54, 50, 58)),
    "infested_stone":   _c((131, 131, 131)),
    "infested_cobblestone":_c((127, 127, 127)),
    "beacon":           _c((120, 214, 208)),
    "lodestone":        _c((130, 132, 138)),
    "target":           _c((214, 188, 174)),
    "dragon_egg":       _c((28, 18, 38)),
    "turtle_egg":       _c((214, 214, 178)),
    "shulker_box":      _c((150, 121, 150)),
    "lantern":          _c((196, 150, 92)),
    "soul_lantern":     _c((96, 168, 188)),
    "bell":             _c((218, 178, 70)),
    "brewing_stand":    _c((130, 110, 96)),
    "flower_pot":       _c((150, 90, 66)),
}

# Wool / concrete / terracotta / glass colour bases by colour word.
DYE_COLORS = {
    "white": (233, 236, 236), "orange": (240, 118, 19), "magenta": (199, 78, 189),
    "light_blue": (58, 175, 217), "yellow": (248, 198, 39), "lime": (112, 185, 25),
    "pink": (237, 141, 172), "gray": (62, 68, 71), "light_gray": (142, 142, 134),
    "cyan": (21, 137, 145), "purple": (121, 42, 172), "blue": (53, 57, 157),
    "brown": (114, 71, 40), "green": (84, 109, 27), "red": (160, 39, 34),
    "black": (25, 26, 29),
}

# Flower / small plant billboard colours.
FLOWER_COLORS = {
    "dandelion": (236, 200, 40), "poppy": (203, 47, 36), "blue_orchid": (47, 165, 220),
    "allium": (160, 96, 200), "azure_bluet": (220, 224, 220), "red_tulip": (200, 44, 40),
    "orange_tulip": (224, 124, 34), "white_tulip": (224, 228, 224), "pink_tulip": (224, 160, 196),
    "oxeye_daisy": (228, 230, 224), "cornflower": (84, 110, 214), "lily_of_the_valley": (236, 240, 236),
    "sunflower": (240, 200, 40), "rose_bush": (190, 44, 40), "peony": (210, 170, 200),
    "lilac": (190, 150, 200), "wither_rose": (40, 40, 40),
    # plain foliage billboards — without these they hit the grey last-resort fallback
    "grass": (96, 165, 60), "fern": (90, 150, 60), "tall_grass": (96, 165, 60),
    "large_fern": (90, 150, 60), "dead_bush": (140, 110, 60), "seagrass": (60, 150, 90),
    "tall_seagrass": (60, 150, 90),
}


# ---------------------------------------------------------------------------
# Block class detection
# ---------------------------------------------------------------------------
def base_name(name: str) -> str:
    n = name.split("[")[0]
    if n.startswith("minecraft:"):
        n = n[len("minecraft:"):]
    return n


# small plant billboards (thin X), upright sprite
PLANT_BILLBOARD = {
    "grass", "fern", "large_fern", "tall_grass", "dead_bush", "seagrass", "tall_seagrass",
    "dandelion", "poppy", "blue_orchid", "allium", "azure_bluet", "red_tulip", "orange_tulip",
    "white_tulip", "pink_tulip", "oxeye_daisy", "cornflower", "lily_of_the_valley",
    "sunflower", "rose_bush", "peony", "lilac", "wither_rose",
    "oak_sapling", "birch_sapling", "spruce_sapling", "jungle_sapling", "acacia_sapling",
    "dark_oak_sapling", "bamboo_sapling", "brown_mushroom", "red_mushroom",
    "nether_wart", "crimson_roots", "warped_roots", "warped_fungus", "crimson_fungus",
    "sweet_berry_bush", "sugar_cane", "bamboo", "kelp", "kelp_plant", "wheat",
    "carrots", "potatoes", "beetroots", "vine", "lily_pad", "cocoa", "chorus_flower",
    "chorus_plant", "cobweb", "sea_pickle",
}

# torch-like: small bright post with a glow tip
TORCH_LIKE = {
    "torch", "wall_torch", "redstone_torch", "redstone_wall_torch",
    "soul_torch", "soul_wall_torch", "end_rod",
}

LEAF_LIKE_SUFFIX = "_leaves"


def get_block_color(name: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Return (top_rgb, side_rgb) for a base block name, never grey-fallback unless truly unknown."""
    b = base_name(name)
    if b in BLOCK_COLORS:
        return BLOCK_COLORS[b]
    if b in FLOWER_COLORS:
        c = FLOWER_COLORS[b]
        return (c, c)

    # dye-coloured families
    for token, fam, sat in (
        ("_wool", DYE_COLORS, 1.0),
        ("_concrete_powder", DYE_COLORS, 0.92),
        ("_concrete", DYE_COLORS, 1.0),
        ("_terracotta", DYE_COLORS, 0.6),
        ("_glazed_terracotta", DYE_COLORS, 0.7),
        ("_stained_glass_pane", DYE_COLORS, 0.85),
        ("_stained_glass", DYE_COLORS, 0.85),
        ("_carpet", DYE_COLORS, 1.0),
        ("_bed", DYE_COLORS, 1.0),
        ("_shulker_box", DYE_COLORS, 0.85),
        ("_banner", DYE_COLORS, 1.0),
        ("_wall_banner", DYE_COLORS, 1.0),
    ):
        if b.endswith(token):
            col = b[: -len(token)]
            if col in fam:
                base = fam[col]
                if token in ("_terracotta",):  # muted/earthy
                    base = tuple(int(0.6 * cc + 0.4 * 120) for cc in base)
                return (base, base)

    # keyword heuristics for the long tail so nothing is pure grey
    def has(*ks):
        return any(k in b for k in ks)

    if b.endswith(LEAF_LIKE_SUFFIX) or "leaves" in b:
        c = (66, 140, 48)
        return (c, c)
    if has("log", "wood", "stem", "stalk"):
        c = (110, 84, 50)
        return (c, (94, 70, 42))
    if has("plank"):
        return ((160, 126, 75), (160, 126, 75))
    if has("slab", "stairs", "wall", "bricks", "brick"):
        for _sp, _wc in (("dark_oak", (86, 62, 40)), ("oak", (160, 126, 75)),
                          ("spruce", (110, 82, 50)), ("birch", (212, 196, 158)),
                          ("jungle", (160, 114, 78)), ("acacia", (176, 98, 60)),
                          ("mangrove", (124, 62, 55)), ("cherry", (225, 160, 165)),
                          ("crimson", (124, 58, 80)), ("warped", (58, 118, 114))):
            if _sp in b:  # wood stairs/slabs/fences must read as wood, not stone-grey
                return (_wc, tuple(int(c * 0.88) for c in _wc))
        if has("nether"):
            return ((52, 28, 32), (52, 28, 32))
        if has("sandstone"):
            return ((216, 203, 157), (216, 203, 157))
        if has("quartz"):
            return ((232, 229, 222), (232, 229, 222))
        if has("blackstone"):
            return ((46, 42, 48), (46, 42, 48))
        if has("prismarine"):
            return ((99, 152, 137), (99, 152, 137))
        return ((125, 122, 117), (125, 122, 117))
    if has("sandstone"):
        return ((216, 203, 157), (216, 203, 157))
    if has("quartz"):
        return ((232, 229, 222), (232, 229, 222))
    if has("prismarine"):
        return ((99, 152, 137), (99, 152, 137))
    if has("coral"):
        if has("dead"):
            return ((125, 120, 112), (125, 120, 112))
        return ((196, 80, 150), (196, 80, 150))
    if has("mushroom_block", "mushroom_stem"):
        return ((220, 212, 198), (220, 212, 198))
    if has("ice"):
        return ((145, 183, 246), (145, 183, 246))
    if has("door", "trapdoor", "fence", "gate", "sign", "ladder", "scaffolding", "barrel",
            "lectern", "loom", "composter", "bookshelf", "table", "chest", "jukebox", "beehive",
            "bee_nest", "campfire"):
        return ((150, 116, 70), (134, 100, 58))
    if has("glass"):
        return ((197, 226, 232), (197, 226, 232))
    if has("rail"):
        return ((140, 120, 90), (140, 120, 90))
    if has("piston", "dispenser", "dropper", "observer", "furnace", "smoker", "blast_furnace",
            "anvil", "grindstone", "stonecutter", "hopper", "cauldron", "iron"):
        return ((118, 118, 120), (108, 108, 110))
    if has("wart"):
        return ((114, 16, 18), (114, 16, 18))
    if has("netherrack", "nether"):
        return ((104, 50, 47), (104, 50, 47))
    if has("end_stone", "end_"):
        return ((221, 223, 165), (221, 223, 165))
    if has("sand"):
        return ((219, 207, 163), (219, 207, 163))
    if has("snow"):
        return ((243, 248, 252), (243, 248, 252))
    if has("water"):
        return ((58, 110, 206), (58, 110, 206))
    if has("lava", "fire", "magma"):
        return ((221, 108, 32), (221, 108, 32))
    if has("nylium"):
        return ((43, 119, 110), (104, 50, 47))
    if has("weeping_vines", "twisting_vines"):
        return ((150, 24, 24), (150, 24, 24))
    if has("redstone"):
        return ((150, 36, 28), (150, 36, 28))
    if has("button", "pressure_plate", "lever", "tripwire", "repeater", "comparator",
            "chain", "bubble_column"):
        return ((120, 118, 118), (110, 108, 108))
    if has("nylium"):
        return ((130, 38, 48), (104, 50, 47))
    if has("vine", "bamboo", "sugar_cane", "sweet_berry", "lily_pad", "kelp", "seagrass",
           "moss", "sapling", "fern", "cactus", "lichen", "sprouts", "roots", "lily"):
        return ((86, 140, 56), (74, 122, 48))   # green foliage, not tan
    # last resort: muted tan rather than pure grey, so it still reads as "a block"
    return ((150, 138, 120), (138, 126, 110))


# ---------------------------------------------------------------------------
# Palette build
# ---------------------------------------------------------------------------
def build_palette(palette_json: str = DEFAULT_PALETTE_JSON):
    """Build per-id render attributes from global_palette.json (name->id)."""
    with open(palette_json) as f:
        name_to_id = json.load(f)

    id_top: Dict[int, Tuple[int, int, int]] = {}
    id_side: Dict[int, Tuple[int, int, int]] = {}
    id_kind: Dict[int, str] = {}     # 'cube' | 'plant' | 'torch' | 'air'
    id_glow: Dict[int, Tuple[int, int, int]] = {}

    for name, cid in name_to_id.items():
        cid = int(cid)
        b = base_name(name)
        if b in AIR_NAMES:
            id_kind[cid] = "air"
            continue
        top, side = get_block_color(name)
        # most-common wins for shared ids: keep first cube assignment but allow plant override
        if cid not in id_kind or id_kind[cid] == "cube":
            id_top[cid] = top
            id_side[cid] = side
        if b in TORCH_LIKE:
            id_kind[cid] = "torch"
            id_glow[cid] = (255, 214, 120) if "soul" not in b else (90, 200, 230)
        elif b in PLANT_BILLBOARD or b.startswith("potted_") or b.endswith("_sapling"):
            id_kind.setdefault(cid, "plant")
            if id_kind[cid] != "torch":
                id_kind[cid] = "plant"
        else:
            id_kind.setdefault(cid, "cube")

    return {
        "top": id_top, "side": id_side, "kind": id_kind, "glow": id_glow,
        "leaf_ids": {cid for name, _id in name_to_id.items()
                     for cid in [int(_id)] if "leaves" in base_name(name)},
    }


# ---------------------------------------------------------------------------
# Shading helpers
# ---------------------------------------------------------------------------
def _shade(rgb, f):
    return tuple(max(0, min(255, int(c * f))) for c in rgb)


# 4x4 Bayer matrix (ordered dither), normalized to ~[-1,1]*amp
_BAYER4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], dtype=np.float32)
_BAYER4 = (_BAYER4 / 15.0 - 0.5)  # in [-0.5, 0.5]


# ---------------------------------------------------------------------------
# Core renderer
# ---------------------------------------------------------------------------
def render_isometric3(
    voxels: np.ndarray,
    palette: dict,
    tile_h: int = TILE_H,
    tile_depth: int = TILE_DEPTH,
    supersample: int = SUPERSAMPLE,
    bg_color=BG_COLOR,
    ao_strength: float = 0.16,
    leaf_alpha: float = 0.82,
    dither_amp: float = 10.0,
) -> Image.Image:
    """Render an [X,Y,Z] integer voxel grid as a high-fidelity isometric PNG."""
    vox = voxels.astype(np.int64)
    Sx, Sy, Sz = vox.shape
    kind = palette["kind"]
    id_top, id_side, id_glow = palette["top"], palette["side"], palette["glow"]
    leaf_ids = palette["leaf_ids"]

    air_ids = {cid for cid, k in kind.items() if k == "air"}

    # solid = blocks that occlude (cubes only; plants/torches do not occlude)
    is_cube = np.zeros(vox.max() + 2, dtype=bool)
    is_air = np.zeros(vox.max() + 2, dtype=bool)
    for cid in range(vox.max() + 1):
        k = kind.get(cid, "cube" if cid not in air_ids else "air")
        if k == "air":
            is_air[cid] = True
        elif k == "cube":
            is_cube[cid] = True
    solid = is_cube[vox]            # occluding cubes
    occupied = ~is_air[vox]         # any non-air (incl. plants/torches)

    # face exposure for cubes
    top_exp = np.zeros_like(solid)
    top_exp[:, :, :-1] = solid[:, :, :-1] & ~solid[:, :, 1:]
    top_exp[:, :, -1] = solid[:, :, -1]
    right_exp = np.zeros_like(solid)
    right_exp[:-1] = solid[:-1] & ~solid[1:]
    right_exp[-1] = solid[-1]
    left_exp = np.zeros_like(solid)
    left_exp[:, :-1] = solid[:, :-1] & ~solid[:, 1:]
    left_exp[:, -1] = solid[:, -1]
    cube_vis = (top_exp | right_exp | left_exp) & solid

    # --- Ambient occlusion: count solid neighbours in a 3x3x3 box (more = darker) ---
    sp = np.pad(solid.astype(np.float32), 1)
    nbr = np.zeros_like(solid, dtype=np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                nbr += sp[1+dx:1+dx+Sx, 1+dy:1+dy+Sy, 1+dz:1+dz+Sz]
    # AO factor in [1-ao_strength, 1]; many neighbours -> darker
    ao = 1.0 - ao_strength * (nbr / 26.0)

    # voxels to draw (cubes with a visible face, plus all plants/torches)
    plant_torch = occupied & ~solid
    draw_mask = cube_vis | plant_torch
    xs, ys, zs = np.where(draw_mask)
    if len(xs) == 0:
        return Image.new("RGB", (256, 256), bg_color)

    th = tile_h * supersample
    td = tile_depth * supersample
    h2 = th // 2

    sxs = (xs - ys) * th
    sys = (xs + ys) * h2 - zs * td
    margin = th * 2
    min_sx = int(sxs.min()) - th - margin
    max_sx = int(sxs.max()) + th + margin
    min_sy = int(sys.min()) - h2 - margin
    max_sy = int(sys.max()) + td + h2 + margin
    img_w = max_sx - min_sx
    img_h = max_sy - min_sy
    off_x, off_y = -min_sx, -min_sy

    img = Image.new("RGB", (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(img, "RGBA")

    # painter's order: bottom layers first, far depth-band first, far cells first
    order = np.lexsort((-ys, xs, xs + ys, zs))
    xs, ys, zs = xs[order], ys[order], zs[order]

    t0 = time.time()
    n = 0
    for i in range(len(xs)):
        x, y, z = int(xs[i]), int(ys[i]), int(zs[i])
        bid = int(vox[x, y, z])
        k = kind.get(bid, "cube")
        cx = (x - y) * th + off_x
        cy = (x + y) * h2 - z * td + off_y

        if k == "plant":
            _draw_plant(draw, cx, cy, th, td, id_top.get(bid, (90, 160, 60)), bid, leaf_ids)
            n += 1
            continue
        if k == "torch":
            _draw_torch(draw, cx, cy, th, td, id_glow.get(bid, (255, 214, 120)))
            n += 1
            continue

        # cube
        top_rgb = id_top.get(bid, (150, 138, 120))
        side_rgb = id_side.get(bid, top_rgb)
        a = float(ao[x, y, z])
        is_leaf = bid in leaf_ids
        alpha = int(255 * leaf_alpha) if is_leaf else 255

        # ordered dither offset per face (texture-ish)
        def dither(px, py):
            return dither_amp * _BAYER4[py & 3, px & 3]

        # LEFT face
        if left_exp[x, y, z]:
            d = dither(x + z, y)
            fc = _shade(side_rgb, SHADE_LEFT * a)
            fc = tuple(max(0, min(255, int(c + d))) for c in fc) + (alpha,)
            draw.polygon([(cx - th, cy), (cx, cy + h2),
                          (cx, cy + h2 + td), (cx - th, cy + td)], fill=fc)
        # RIGHT face
        if right_exp[x, y, z]:
            d = dither(y + z, x)
            fc = _shade(side_rgb, SHADE_RIGHT * a)
            fc = tuple(max(0, min(255, int(c + d))) for c in fc) + (alpha,)
            draw.polygon([(cx + th, cy), (cx, cy + h2),
                          (cx, cy + h2 + td), (cx + th, cy + td)], fill=fc)
        # TOP face
        if top_exp[x, y, z]:
            d = dither(x, y)
            fc = _shade(top_rgb, SHADE_TOP * a)
            fc = tuple(max(0, min(255, int(c + d))) for c in fc) + (alpha,)
            draw.polygon([(cx, cy - h2), (cx + th, cy),
                          (cx, cy + h2), (cx - th, cy)], fill=fc)
        n += 1

    img = _autocrop(img, bg_color, margin=8 * supersample)
    if supersample > 1:
        img = img.resize((max(1, img.width // supersample),
                          max(1, img.height // supersample)),
                         Image.LANCZOS)
    dt = time.time() - t0
    print(f"[render3] drew {n} voxels in {dt:.2f}s -> {img.size[0]}x{img.size[1]}")
    return img


def _draw_plant(draw, cx, cy, th, td, color, bid, leaf_ids):
    """Thin X-shaped billboard standing on the voxel cell center."""
    c = color + (235,)
    stem = tuple(int(0.6 * a + 0.4 * b) for a, b in zip(color, (70, 120, 50))) + (235,)
    base_y = cy + th // 2          # bottom (ground) of the cell
    top_y = base_y - int(td * 1.15)
    midx = cx
    w = int(th * 0.42)
    lw = max(1, th // 5)
    # two crossed blades
    draw.line([(midx - w, base_y), (midx + int(w * 0.5), top_y)], fill=stem, width=lw)
    draw.line([(midx + w, base_y), (midx - int(w * 0.5), top_y)], fill=stem, width=lw)
    # flower head / tip blob if it's a coloured flower (not plain grass/fern)
    if max(color) - min(color) > 40 or sum(color) > 560:
        r = max(2, th // 4)
        draw.ellipse([midx - r, top_y - r, midx + r, top_y + r], fill=c)


def _draw_torch(draw, cx, cy, th, td, glow):
    """Small post + glowing tip."""
    base_y = cy + th // 2
    top_y = base_y - int(td * 1.0)
    wood = (140, 105, 60, 255)
    lw = max(1, th // 4)
    draw.line([(cx, base_y), (cx, top_y)], fill=wood, width=lw)
    r = max(2, th // 4)
    draw.ellipse([cx - r, top_y - r, cx + r, top_y + r], fill=glow + (255,))
    # soft halo
    r2 = r * 2
    draw.ellipse([cx - r2, top_y - r2, cx + r2, top_y + r2], fill=glow + (60,))


def _autocrop(img, bg_color, margin=16):
    arr = np.array(img)
    bg = np.array(bg_color, dtype=np.uint8)
    mask = np.any(np.abs(arr.astype(int)[:, :, :3] - bg) > 4, axis=2)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - margin); rmax = min(arr.shape[0], rmax + margin + 1)
    cmin = max(0, cmin - margin); cmax = min(arr.shape[1], cmax + margin + 1)
    return img.crop((cmin, rmin, cmax, rmax))


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def render_voxel_grid(voxels, palette_json=DEFAULT_PALETTE_JSON, swap_yz=True, **kw):
    pal = build_palette(palette_json)
    if swap_yz:
        voxels = np.swapaxes(voxels, 1, 2)
    return render_isometric3(voxels, pal, **kw)


def _load_voxels(path, key, frame, batch):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        v = np.load(path)
    elif ext == ".npz":
        d = np.load(path)
        v = d[key]
    elif ext == ".pt":
        import torch
        d = torch.load(path, map_location="cpu")
        v = d[key] if isinstance(d, dict) else d
        v = v.numpy() if hasattr(v, "numpy") else np.array(v)
    else:
        raise ValueError(f"unknown ext {ext}")
    while v.ndim > 3:
        if v.ndim == 5:
            v = v[batch]
        elif v.ndim == 4:
            v = v[frame]
    return v.astype(np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--voxels", required=True)
    p.add_argument("--key", default="grids")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--output", default="iso3.png")
    p.add_argument("--palette_json", default=DEFAULT_PALETTE_JSON)
    p.add_argument("--tile_h", type=int, default=TILE_H)
    p.add_argument("--tile_depth", type=int, default=TILE_DEPTH)
    p.add_argument("--supersample", type=int, default=SUPERSAMPLE)
    p.add_argument("--no_swap_yz", action="store_true",
                   help="disable the Y-up swapaxes(1,2) (enabled by default)")
    args = p.parse_args()

    v = _load_voxels(args.voxels, args.key, args.frame, args.batch)
    print(f"[render3] loaded {v.shape} {v.dtype}")
    pal = build_palette(args.palette_json)
    if not args.no_swap_yz:
        v = np.swapaxes(v, 1, 2)
    img = render_isometric3(v, pal, tile_h=args.tile_h, tile_depth=args.tile_depth,
                            supersample=args.supersample)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    img.save(args.output)
    print(f"[render3] saved {args.output} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
