# paint_config.py
from typing import NamedTuple, Sequence, Tuple, Optional

class MusclePair(NamedTuple):
    name: str
    upper: Sequence[int]
    lower: Sequence[int]
    color: Tuple[int,int,int] = (0,0,255)
    core_alpha: float = 0.60
    edge_alpha: float = 0.38
    inner_w: float = 10
    outer_w: float = 18
    samples: int = 16
    feather: int = 4
    # Optional polygon "hole" (unfilled region)
    upper_unfilled: Optional[Sequence[int]] = None
    lower_unfilled: Optional[Sequence[int]] = None
    cut_type: str = "poly"
    cut_feather_px: float = 3.0
    cut_dilate_px: float = 0.0
    cut_use_catmull: bool = True
    cut_samples_per_seg: int = 12

ALL_MUSCLE_PAIRS = [
    MusclePair("Left_inner_brow_raiser", upper=(69, 108), lower=(65, 55) ),
    MusclePair("Left_outer_brow_raiser", upper=(71, 104), lower=(139, 53), core_alpha=0.55, outer_w=14),
    MusclePair("Left_brow_lowerer",      upper=(107, 9),  lower=(189, 168), core_alpha=0.55, outer_w=1),
    MusclePair("Left_upper_lid_raiser",  upper=(224, 222),lower=(29, 28),   core_alpha=0.55, outer_w=10),
    MusclePair("Left_cheek raiser",      upper=(233, 231, 230, 228, 111), lower=(47, 100, 101, 50, 111), core_alpha=0.55, outer_w=10),
    MusclePair("Left_lid tighter",
        upper=(35, 30, 28, 190, 243), lower=(35, 228, 230, 232, 243),
        core_alpha=0.55, outer_w=10,
        upper_unfilled=(130, 160, 158, 157, 173),
        lower_unfilled=(7, 163, 472, 153, 154),
        cut_type="poly", cut_feather_px=2.5, cut_dilate_px=0.0, cut_use_catmull=True, cut_samples_per_seg=14
    ),
    MusclePair("Left_nose_wrinkler",     upper=(100, 114, 236), lower=(203, 220, 236), core_alpha=0.55, outer_w=10),
    MusclePair("Left_upper_lip_raiser",  upper=(206, 203), lower=(185, 39), core_alpha=0.55, outer_w=10),
    MusclePair("Left_Lip_corner_puller", upper=(147, 123), lower=(57, 40),  core_alpha=0.55, outer_w=10),
    MusclePair("Left_dimpler",           upper=(192, 187), lower=(214, 212, 185), core_alpha=0.55, outer_w=10),
    MusclePair("Left_Lip_corner_depressor", upper=(136,135,57), lower=(149, 204, 61), core_alpha=0.55, outer_w=10),
    MusclePair("Left_chin_raiser",       upper=(91,84,83), lower=(106,140,171,208,201), core_alpha=0.55, outer_w=10),
    MusclePair("Left_lip_stretcher",     upper=(58,192,186,61), lower=(214,212,57,61), core_alpha=0.55, outer_w=10),
    MusclePair("Left_lip_tightener",
        upper=(57,92,167,393,391,410,287),
        lower=(57,43,83,313,406,335,273,287),
        core_alpha=0.55, outer_w=10,
        upper_unfilled=(78, 41, 12, 268, 271, 304),
        lower_unfilled=(78,179,14,316,403,319),
        cut_type="poly", cut_feather_px=2.5, cut_dilate_px=0.0, cut_use_catmull=True, cut_samples_per_seg=2
    ),
    MusclePair("Left_lips_part",         upper=(91,17), lower=(150,140,201,200), core_alpha=0.55, outer_w=10),
    MusclePair("Left_jaw_drop",          upper=(93,123,187,207), lower=(58,172,136), core_alpha=0.55, outer_w=10),

    MusclePair("Right_inner_brow_raiser", upper=(337, 299), lower=(285, 295), core_alpha=0.55, outer_w=14),
    MusclePair("Right_outer_brow_raiser", upper=(333, 300), lower=(282, 276), core_alpha=0.55, outer_w=14),
    MusclePair("Right_brow_lowerer",      upper=(9, 336),   lower=(168, 413), core_alpha=0.55, outer_w=1),
    MusclePair("Right_upper_lid_raiser",  upper=(442, 444), lower=(258, 259), core_alpha=0.55, outer_w=10),
    MusclePair("Right_cheek raiser",      upper=(453,451,450,448,340), lower=(277,329,330,280,340), core_alpha=0.55, outer_w=10),
    MusclePair("Right_lid tighter",
        upper=(464, 413, 258, 259, 446), lower=(464, 452, 450, 448, 446),
        core_alpha=0.55, outer_w=10,
        upper_unfilled=(398,384,385,387,359),
        lower_unfilled=(381,380,374,390,249),
        cut_type="poly", cut_feather_px=2.5, cut_dilate_px=0.0, cut_use_catmull=True, cut_samples_per_seg=14
    ),
    MusclePair("Right_nose_wrinkler",     upper=(456,343,329), lower=(456,440,423), core_alpha=0.55, outer_w=10),
    MusclePair("Right_upper_lip_raiser",  upper=(423,426), lower=(269,409), core_alpha=0.55, outer_w=10),
    MusclePair("Right_Lip_corner_puller", upper=(352,376), lower=(270,287), core_alpha=0.55, outer_w=10),
    MusclePair("Right_dimpler",           upper=(411,416), lower=(409,432,434), core_alpha=0.55, outer_w=10),
    MusclePair("Right_Lip_corner_depressor", upper=(287,365,379), lower=(375,424,378), core_alpha=0.55, outer_w=10),
    MusclePair("Right_chin_raiser",       upper=(313,314,321), lower=(421,428,396,369,335), core_alpha=0.55, outer_w=10),
    MusclePair("Right_lip_stretcher",     upper=(409,410,416,288), lower=(409,287,432,434), core_alpha=0.55, outer_w=10),
    MusclePair("Right_lips_part",         upper=(17,321), lower=(200,421,369,379), core_alpha=0.55, outer_w=10),
    MusclePair("Right_jaw_drop",          upper=(427,411,352,323), lower=(365,397,288), core_alpha=0.55, outer_w=10),


    MusclePair("Upper_lip_pucker",    upper=(37, 267), lower=(82, 312), core_alpha=0.55, outer_w=10),
    MusclePair("Lower_lip_pucker",    upper=(86, 316), lower=(84, 314), core_alpha=0.55, outer_w=10),


]


# Map AU codes (strings) to the muscle pair names to draw when active.
# Examples requested:
#  - AU1 (Inner brow raiser) -> Left/Right inner brow raisers
#  - AU18 (Lip pucker)       -> Upper/Lower lip pucker
AU_TO_MUSCLES = {'1': ['Left_inner_brow_raiser', 'Right_inner_brow_raiser'],
 '2': ['Left_outer_brow_raiser', 'Right_outer_brow_raiser'],
 '4': ['Left_brow_lowerer', 'Right_brow_lowerer'],
 '5': ['Left_upper_lid_raiser', 'Right_upper_lid_raiser'],
 '6': ['Left_cheek raiser', 'Right_cheek raiser'],
 '7': ['Left_lid tighter', 'Right_lid tighter'],
 '9': ['Left_nose_wrinkler', 'Right_nose_wrinkler'],
 '10': ['Left_upper_lip_raiser', 'Right_upper_lip_raiser'],
 '12': ['Left_Lip_corner_puller', 'Right_Lip_corner_puller'],
 '14': ['Left_dimpler', 'Right_dimpler'],
 '15': ['Left_Lip_corner_depressor', 'Right_Lip_corner_depressor'],
 '17': ['Left_chin_raiser', 'Right_chin_raiser'],
 '20': ['Left_lip_stretcher', 'Right_lip_stretcher'],
 '23': ['Left_lip_tightener'],
 '25': ['Left_lips_part', 'Right_lips_part'],
 '26': ['Left_jaw_drop', 'Right_jaw_drop'],
 '18': ['Upper_lip_pucker', 'Lower_lip_pucker']
 }

# Convenience: quick lookup by name -> MusclePair
MUSCLE_BY_NAME = {mp.name: mp for mp in ALL_MUSCLE_PAIRS}


# ---- Auto-build AU_TO_MUSCLES from names ------------------------------------
# import re
# from collections import defaultdict
# from typing import Dict, List
# from au_config import AU_INDEX, AU_NAMES  # uses canonical AU names you already have

# def _norm(s: str) -> str:
#     s = s.lower().replace("’", "'")
#     s = re.sub(r"[_\-]+", " ", s)          # underscores/dashes -> space
#     s = re.sub(r"[^a-z0-9]+", " ", s)      # non-alnum -> space
#     s = re.sub(r"\s+", " ", s).strip()
#     return s

# # Build a canonical "AU name phrase" -> AU code map from au_config.
# # e.g. "inner brow raiser" -> "1"
# _NAMEPHRASE_TO_CODE: Dict[str, str] = {}
# for code, name in zip(AU_INDEX, AU_NAMES):
#     _NAMEPHRASE_TO_CODE[_norm(name)] = code

# # Minimal synonyms/variants you use in muscle names -> canonical AU phrase
# # (Extend if you introduce new naming variants.)
# _SYNONYMS_TO_CANON = {
#     "lid tighter":            "lid tightener",       # AU7
#     "lip corner puller":      "lip corner puller",   # AU12 (kept for clarity)
#     "lip corner depressor":   "lip corner depressor",# AU15
#     "inner brow raiser":      "inner brow raiser",   # AU1
#     "outer brow raiser":      "outer brow raiser",   # AU2
#     "brow lowerer":           "brow lowerer",        # AU4
#     "upper lid raiser":       "upper lid raiser",    # AU5
#     "cheek raiser":           "cheek raiser",        # AU6
#     "nose wrinkler":          "nose wrinkler",       # AU9
#     "upper lip raiser":       "upper lip raiser",    # AU10
#     "dimpler":                "dimpler",             # AU14
#     "chin raiser":            "chin raiser",         # AU17
#     "lip pucker":             "lip pucker",          # AU18
#     "lip stretcher":          "lip stretcher",       # AU20
#     "lip tightener":          "lip tightener",       # AU23
#     "lip pressor":            "lip pressor",         # AU24
#     "lips part":              "lips part",           # AU25
#     "jaw drop":               "jaw drop",            # AU26
#     "mouth stretch":          "mouth stretch",       # AU27
#     # Optional (present in AU_NAMES but may not exist in muscles):
#     "nasolabial deepener":    "nasolabial deepener", # AU11
#     "sharp lip puller":       "sharp lip puller",    # AU13
#     "lower lip depressor":    "lower lip depressor", # AU16
#     "tongue show":            "tongue show",         # AU19
#     "lip funneler":           "lip funneler",        # AU22
# }

# # Convert synonyms -> AU code (only if that AU exists in AU_NAMES)
# _KEYWORD_TO_CODE: Dict[str, str] = {}
# for syn_phrase, canon_phrase in _SYNONYMS_TO_CANON.items():
#     canon_norm = _norm(canon_phrase)
#     if canon_norm in _NAMEPHRASE_TO_CODE:
#         _KEYWORD_TO_CODE[_norm(syn_phrase)] = _NAMEPHRASE_TO_CODE[canon_norm]

# def _best_keyword(norm_name: str) -> str | None:
#     """
#     Return the most specific matching AU keyword contained in norm_name.
#     Uses longest-match wins to avoid e.g. 'brow raiser' stealing from 'inner brow raiser'.
#     """
#     matches = [(kw, len(kw)) for kw in _KEYWORD_TO_CODE.keys() if kw in norm_name]
#     if not matches:
#         return None
#     matches.sort(key=lambda t: t[1], reverse=True)
#     return matches[0][0]

# def build_au_to_muscles(pairs=ALL_MUSCLE_PAIRS) -> Dict[str, List[str]]:
#     out: Dict[str, List[str]] = defaultdict(list)
#     for mp in pairs:
#         nm = _norm(mp.name)
#         kw = _best_keyword(nm)
#         if kw is None:
#             continue
#         code = _KEYWORD_TO_CODE[kw]
#         out[code].append(mp.name)
#     return dict(out)

# # Public objects used by draw_utils.py
# AU_TO_MUSCLES: Dict[str, List[str]] = build_au_to_muscles()
# MUSCLE_BY_NAME = {mp.name: mp for mp in ALL_MUSCLE_PAIRS}

# # Optional: quick sanity print (disable in production)
# if __name__ == "__main__":
#     import pprint
#     pprint.pp(AU_TO_MUSCLES)


