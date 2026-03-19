"""
478 MediaPipe dense landmarks → 68 dlib sparse landmarks conversion.

Ported from: flame-head-tracker/utils/mp2dlib.py
Original author: Peizhi Yan
Source: https://github.com/PeizhiYan/Mediapipe_2_Dlib_Landmarks
"""

import numpy as np

# Index correspondences: each entry maps a dlib landmark (1-68)
# to one or two MediaPipe indices. Two indices → average their coords.
mp2dlib_correspondence = [
    # Face Contour (1-17)
    [127],       # 1
    [234],       # 2
    [93],        # 3
    [132, 58],   # 4
    [58, 172],   # 5
    [136],       # 6
    [150],       # 7
    [176],       # 8
    [152],       # 9
    [400],       # 10
    [379],       # 11
    [365],       # 12
    [397, 288],  # 13
    [361],       # 14
    [323],       # 15
    [454],       # 16
    [356],       # 17
    # Right Brow (18-22)
    [156],       # 18
    [70, 63],    # 19
    [105],       # 20
    [66],        # 21
    [107, 55],   # 22
    # Left Brow (23-27)
    [336, 285],  # 23
    [296],       # 24
    [334],       # 25
    [293, 300],  # 26
    [383],       # 27
    # Nose (28-36)
    [168, 6],    # 28
    [197, 195],  # 29
    [5],         # 30
    [4],         # 31
    [98],        # 32
    [97],        # 33
    [2],         # 34
    [326],       # 35
    [327],       # 36
    # Right Eye (37-42)
    [33],        # 37
    [160],       # 38
    [158],       # 39
    [133],       # 40
    [153],       # 41
    [144],       # 42
    # Left Eye (43-48)
    [362],       # 43
    [385],       # 44
    [387],       # 45
    [263],       # 46
    [373],       # 47
    [380],       # 48
    # Upper Lip Contour Top (49-55)
    [61],        # 49
    [39],        # 50
    [37],        # 51
    [0],         # 52
    [267],       # 53
    [269],       # 54
    [291],       # 55
    # Lower Lip Contour Bottom (56-60)
    [321],       # 56
    [314],       # 57
    [17],        # 58
    [84],        # 59
    [91],        # 60
    # Upper Lip Contour Bottom (61-65)
    [78],        # 61
    [82],        # 62
    [13],        # 63
    [312],       # 64
    [308],       # 65
    # Lower Lip Contour Top (66-68)
    [317],       # 66
    [14],        # 67
    [87],        # 68
]

# Pad single-index entries to pairs for uniform indexing
for ri in range(68):
    if len(mp2dlib_correspondence[ri]) == 1:
        idx = mp2dlib_correspondence[ri][0]
        mp2dlib_correspondence[ri] = [idx, idx]


def convert_landmarks_mediapipe_to_dlib(lmks_mp: np.ndarray) -> np.ndarray:
    """Convert 478 MediaPipe landmarks to 68 dlib landmarks.

    Args:
        lmks_mp: MediaPipe landmarks [478, 2] or [478, 3]

    Returns:
        Converted dlib landmarks [68, 2] or [68, 3]
    """
    return lmks_mp[mp2dlib_correspondence].mean(axis=1)
