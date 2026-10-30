"""
Deletion templates of the 12-subiteration curve thinning algorithm.

K. Palagyi and A. Kuba, A Parallel 3D 12-Subiteration Thinning Algorithm,
Graphical Models and Image Processing 61, 199-221 (1999).

TEMPLATES holds T1 to T14, the set T_US of the first subiteration (deletion direction US,
Fig. 3). Positions use the indices of Fig. 8a, where 1 is the central point p, seen from
above with north at the top:

    layer U        middle layer     layer D
     8  9 10       16 17 18         25 26 27      N
     5  6  7       14  1 15         22 23 24
     2  3  4       11 12 13         19 20 21      S
     W     E

Each template lists its black and white positions and its groups:
    ("x", {...})   at least one of the positions is black
    ("v", {...})   at least one of the positions is white (the "v" and "w" marks)
    ("z", a, b)    positions a and b differ
Positions not listed are "don't care". The central point is black in every template.
"""
import itertools

# Fig. 8a: index by (layer U to D, row S to N, column W to E)
LAYOUT = (((2, 3, 4), (5, 6, 7), (8, 9, 10)),
          ((11, 12, 13), (14, 1, 15), (16, 17, 18)),
          ((19, 20, 21), (22, 23, 24), (25, 26, 27)))

# Lookup table bit k belongs to the k-th (u, n, e) offset in this order (central point skipped)
BIT_OFFSETS = tuple(o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0))


def offset(index):
    """(u, n, e) offset of a Fig. 8a position from p: up, north and east count as +1."""
    for layer, rows in enumerate(LAYOUT):
        for row, columns in enumerate(rows):
            if index in columns:
                return (1 - layer, row - 1, columns.index(index) - 1)
    raise ValueError(f"Fig. 8a has no position {index}")


TEMPLATES = {
    # T1, Fig. 3
    "T1": {"black": {23},
           "white": {2, 3, 4, 5, 6, 7, 8, 9, 10},
           "groups": [("x", {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27})]},
    # T2, Fig. 3
    "T2": {"black": {17},
           "white": {2, 3, 4, 11, 12, 13, 19, 20, 21},
           "groups": [("x", {5, 6, 7, 8, 9, 10, 14, 15, 16, 18, 22, 23, 24, 25, 26, 27})]},
    # T3, Fig. 3
    "T3": {"black": {26},
           "white": {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 19, 20, 21},
           "groups": [("x", {14, 15, 16, 18, 22, 24, 25, 27})]},
    # T4, Fig. 3
    "T4": {"black": {17, 23},
           "white": {2, 3, 4, 6, 12},
           "groups": [("v", {5, 11}), ("v", {7, 13})]},
    # T5, Fig. 3 (also Fig. 8b and the formula g5 in Fig. 8)
    "T5": {"black": {4, 17, 23},
           "white": {2, 3, 6, 12},
           "groups": [("v", {5, 11}), ("z", 7, 13)]},
    # T6, Fig. 3
    "T6": {"black": {2, 17, 23},
           "white": {3, 4, 6, 12},
           "groups": [("v", {7, 13}), ("z", 5, 11)]},
    # T7, Fig. 3
    "T7": {"black": {15, 17, 23},
           "white": {2, 3, 6, 12},
           "groups": [("v", {5, 11})]},
    # T8, Fig. 3
    "T8": {"black": {14, 17, 23},
           "white": {3, 4, 6, 12},
           "groups": [("v", {7, 13})]},
    # T9, Fig. 3
    "T9": {"black": {2, 15, 17, 23},
           "white": {3, 6, 12},
           "groups": [("z", 5, 11)]},
    # T10, Fig. 3
    "T10": {"black": {4, 14, 17, 23},
            "white": {3, 6, 12},
            "groups": [("z", 7, 13)]},
    # T11, Fig. 3
    "T11": {"black": {24, 26},
            "white": {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 19, 20},
            "groups": []},
    # T12, Fig. 3
    "T12": {"black": {22, 26},
            "white": {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 20, 21},
            "groups": []},
    # T13, Fig. 3
    "T13": {"black": {18, 26},
            "white": {2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 19, 20, 21},
            "groups": []},
    # T14, Fig. 3
    "T14": {"black": {16, 26},
            "white": {2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 19, 20, 21},
            "groups": []},
}
