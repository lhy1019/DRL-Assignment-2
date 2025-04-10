import numpy as np
from typing import List, Tuple
from custom2048 import Game2048

# Constants that you can adapt to your environment:
SIZE = 4          # Board dimension, e.g., 4x4 in 2048
MAX_EXPONENT = 15 # Highest exponent you expect to see. For example:
                  # a tile of value 2^16 = 65536. Adjust as needed.

# A Move can be an integer 0..3 or an enum. We'll just use int for demo.
# e.g. 0=Up, 1=Right, 2=Down, 3=Left

###############################################################################
# Helper Functions
###############################################################################

def transform_coord(r: int, c: int, t: int) -> Tuple[int,int]:
    """
    Apply one of eight transformations to row,col:
      t = 0: rotate 0 deg
      t = 1: rotate 90 deg
      t = 2: rotate 180 deg
      t = 3: rotate 270 deg
      t = 4: reflect horizontally
      t = 5: reflect vertically
      t = 6: reflect main diagonal
      t = 7: reflect anti-diagonal
    """
    if t == 0:  # rotate 0 deg
        nr, nc = r, c
    elif t == 1:  # rotate 90 deg
        nr, nc = c, SIZE - 1 - r
    elif t == 2:  # rotate 180 deg
        nr, nc = SIZE - 1 - r, SIZE - 1 - c
    elif t == 3:  # rotate 270 deg
        nr, nc = SIZE - 1 - c, r
    elif t == 4:  # reflect horizontal
        nr, nc = r, SIZE - 1 - c
    elif t == 5:  # reflect vertical
        nr, nc = SIZE - 1 - r, c
    elif t == 6:  # reflect main diagonal
        nr, nc = c, r
    elif t == 7:  # reflect anti-diagonal
        nr, nc = SIZE - 1 - c, SIZE - 1 - r
    else:
        # fallback, shouldn't happen if t is in [0..7]
        nr, nc = r, c
    return nr, nc

def tile_to_exponent(tile_value: int) -> int:
    """
    Convert a tile's numeric value to its exponent.
      tile_value  = 0  => exponent = 0
      tile_value  = 2  => exponent = 1
      tile_value  = 4  => exponent = 2
      tile_value  = 8  => exponent = 3
      ...
    If tile_value is not a power of two, adjust or clamp as you see fit.
    """
    if tile_value < 2:
        return 0
    exp = 0
    val = 1
    while val < tile_value:
        val <<= 1
        exp += 1
    return exp

def pattern_index(grid, pattern_coords: List[Tuple[int,int]]) -> int:
    """
    Given a list of (row,col) coords (the "pattern"), 
    compute the index by treating each tile exponent as a digit 
    in base = (MAX_EXPONENT+1).
    """
    base = MAX_EXPONENT + 1
    idx = 0
    for (r, c) in pattern_coords:
        exp = tile_to_exponent(grid[r][c])
        idx = idx * base + exp
    return idx

###############################################################################
# Data Structures
###############################################################################

class Pattern:
    """
    A pattern is simply a list of (row, col) positions to be observed.
    """
    def __init__(self, coords: List[Tuple[int,int]]):
        self.coords = coords

def lut_size(pattern_length: int) -> int:
    """
    The LUT size for one pattern, if you store all possible exponent
    combinations in a big array. That is: (MAX_EXPONENT+1)^pattern_length.
    """
    return (MAX_EXPONENT + 1) ** pattern_length


class NTupleAgent:
    """
    Python version of the N-tuple Agent that keeps multiple patterns,
    each with a large LUT of weights. 
    """
    def __init__(self, patterns: List[Pattern]):
        self.patterns = patterns

        # Instead of an unordered_map, we can store everything
        # in a list (or dict). We do the big array approach:
        self.weights_arrays = []  # list of lists, one array per pattern

        # Initialize lookups:
        self.setup_lut()

    def setup_lut(self):
        self.weights_arrays = []
        for pat in self.patterns:
            pat_len = len(pat.coords)
            size_needed = (MAX_EXPONENT + 1) ** pat_len
            # Create a float32 array of zeros
            w = np.zeros(size_needed, dtype=np.float32)
            self.weights_arrays.append(w)
            
    def transform_index(self, board, pattern: Pattern, t: int) -> int:
        """
        Compute the LUT index after applying transformation 't'.
        This is essentially the same logic as pattern_index,
        but using transform_coord on each (row,col) of the pattern.
        """
        base = MAX_EXPONENT + 1
        idx = 0
        for (r, c) in pattern.coords:
            nr, nc = transform_coord(r, c, t)
            tile_val = board.grid[nr][nc]
            e = tile_to_exponent(tile_val)
            idx = idx * base + e
        return idx

    def value(self, board) -> float:
        """
        Compute the value estimate of a given board by summing
        the pattern lookups over each pattern and each transformation.
        """
        sum_v = 0.0
        for pIndex, pattern in enumerate(self.patterns):
            w_array = self.weights_arrays[pIndex]
            for t in range(8):
                idx = self.transform_index(board, pattern, t)
                sum_v += w_array[idx]
        return sum_v

    def update(self, board, delta: float, alpha: float):
        """
        Adjust weights by alpha * delta for each pattern + transformation.
        Optionally dividing the delta among the 8 transformations
        so that the total effect is delta (this is a design choice).
        """
        for pIndex, pattern in enumerate(self.patterns):
            w_array = self.weights_arrays[pIndex]
            for t in range(8):
                idx = self.transform_index(board, pattern, t)
                # If you want to distribute the update among the symmetries:
                w_array[idx] += (alpha * delta / 8.0)
                # If you prefer not to split the delta, do:
                # w_array[idx] += alpha * delta
                
    def load_weights(self, weights: np.ndarray):
        """
        Load weights from a flat NumPy array. Assumes the array 'weights'
        is in the same order as the patterns were created.
        """
        offset = 0
        for pIndex, pattern in enumerate(self.patterns):
            pat_len = len(pattern.coords)
            size_needed = (MAX_EXPONENT + 1) ** pat_len

            # Here, we slice out the portion of 'weights' for this pattern.
            # Slicing a NumPy array returns a "view" (no copy),
            # which we can assign into self.weights_arrays[pIndex].
            self.weights_arrays[pIndex] = weights[offset : offset + size_needed]
            
            offset += size_needed


    def best_action(self, board) -> int:
        """
        Evaluate each legal move's afterstate, pick the move with the best value.
        env is an object that can check isMoveLegal(a) and simulateAfterstate(a).
        Returns the best move's index (0..3). If no move is legal, returns -1.
        """
        env = Game2048()
        env.reset(board)
        best_val = -1e15
        best_act = -1
        for a in range(4):
            if env.isMoveLegal(a):
                after_board = env.simulateAfterstate(a)
                v = self.value(after_board)
                if v > best_val:
                    best_val = v
                    best_act = a
        env = None
        return best_act
