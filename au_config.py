# au_config.py
# AU indices, names, thresholds and which AUs to display

# Which AU logits (by model output index) to show
INDEX_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 14, 15, 17, 19, 21, 22]

# Per-AU trigger thresholds (same length as INDEX_LIST)
THRESHOLDS = [0.6, 0.4, 0.5, 0.42, 0.25, 0.85, 0.26, 0.4, 0.5,
              0.45, 0.13, 0.42, 0.20, 0.29, 0.20, 0.89, 0.30]

# FACS codes & names (longer master lists)
AU_INDEX = [
    '1','2','4','5','6','7','9','10','11','12','13','14','15','16','17',
    '18','19','20','22','23','24','25','26','27','32','38','39',
    'L1','R1','L2','R2','L4','R4','L6','R6','L10','R10','L12','R12','L14','R14'
]

AU_NAMES = [
    'Inner brow raiser','Outer brow raiser','Brow lowerer','Upper lid raiser',
    'Cheek raiser','Lid tightener','Nose wrinkler','Upper lip raiser',
    'Nasolabial deepener','Lip corner puller','Sharp lip puller','Dimpler',
    'Lip corner depressor','Lower lip depressor','Chin raiser','Lip pucker',
    'Tongue show','Lip stretcher','Lip funneler','Lip tightener',
    'Lip pressor','Lips part','Jaw drop','Mouth stretch','Lip bite',
    'Nostril dilator','Nostril compressor'
]

# ---- Helpers to look up indices / thresholds quickly ----
# AU code (e.g., "1","18") -> model output index
CODE_TO_IDX = {code: i for i, code in enumerate(AU_INDEX)}

# index -> threshold (only for those in INDEX_LIST)
THRESH_BY_IDX = {idx: thr for idx, thr in zip(INDEX_LIST, THRESHOLDS)}

# AU code -> threshold (only for those in INDEX_LIST)
THRESH_BY_CODE = {AU_INDEX[idx]: THRESH_BY_IDX[idx] for idx in INDEX_LIST if idx < len(AU_INDEX)}
