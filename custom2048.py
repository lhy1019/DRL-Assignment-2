import random
import time
SIZE = 4
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3 
class Board:
    """
    Python equivalent of the C++ struct Board {
        int grid[SIZE][SIZE];
        int score;
        bool terminal;
    }
    """
    def __init__(self, grid=None, score=0, terminal=False):
        if grid is None:
            grid = [[0]*SIZE for _ in range(SIZE)]
        # Make a deep copy to be safe:
        self.grid = [row[:] for row in grid]
        self.score = score
        self.terminal = terminal

    def copy(self):
        return Board(self.grid, self.score, self.terminal)
    
    def reset(self, user_board=None):
        """
        Reset the board to a new state. If user_board is provided, use it.
        """
        if user_board is not None:          
            self.grid = user_board.grid
            self.score = user_board.score
            self.terminal = user_board.terminal
        else:
            # Reset to a new empty board
            self.grid = [[0]*SIZE for _ in range(SIZE)]
            self.score = 0
            self.terminal = False
    
class Game2048:
    def __init__(self):
        # You can seed however you like. Here we just seed with current time:
        random.seed(time.time_ns())
        self.board = Board()
        self.reset()

    def reset(self, user_board: Board = None):
        """
        Start a fresh game: 2 tiles on board, score=0, not terminal
        If user_board is provided, use it directly.
        """
        if user_board is not None:
            self.board = user_board.copy()
            # Optionally recalculate or trust user_board's 'terminal' state:
            self.board.terminal = self.check_game_over()
        else:
            self.board = Board()
            self.board.terminal = False
            self.board.score = 0
            # spawn two tiles
            self.spawn_tile()
            self.spawn_tile()

    def getBoard(self) -> Board:
        """Return a copy of the current board state."""
        return self.board.copy()

    def isTerminal(self) -> bool:
        return self.board.terminal

    def getScore(self) -> int:
        return self.board.score

    def isMoveLegal(self, move: int) -> bool:
        """
        Return True if the given move can actually shift/merge any tile.
        We simulate the afterstate and compare with the original board.
        """
        after = self.simulateAfterstate(move)
        # If grids differ, then the move was effective
        for r in range(SIZE):
            for c in range(SIZE):
                if after.grid[r][c] != self.board.grid[r][c]:
                    return True
        return False

    def step(self, move: int) -> int:
        """
        Execute one environment step with the given move.
         1) Compute the afterstate (slide/merge).
         2) If no change, move is illegal -> reward = 0, no spawn
         3) Otherwise, update board, spawn tile, return merge reward
        """
        if self.board.terminal:
            return 0

        after = self.simulateAfterstate(move)
        # check if anything changed
        changed = False
        for r in range(SIZE):
            for c in range(SIZE):
                if after.grid[r][c] != self.board.grid[r][c]:
                    changed = True
                    break
            if changed:
                break

        if not changed:
            # No tile moved -> illegal move
            print("Illegal move!")
            return 0

        # A valid move => reward from merges is difference in after.score vs current
        reward = after.score - self.board.score
        self.board = after
        # Now spawn a new tile
        self.spawn_tile()
        return reward

    def simulateAfterstate(self, move: int) -> Board:
        """
        Return a Board representing the state after applying 'move'
        (slide & merge) but *without* spawning a new tile.
        """
        bcopy = self.board.copy()
        old_score = bcopy.score

        # Rotate/transform so that 'move' is effectively a LEFT move:
        self._rotate_board_to_left(bcopy, move)

        # Slide & merge left
        merge_reward = self._slide_left_and_merge(bcopy)
        bcopy.score = old_score + merge_reward

        # Rotate/transform back
        self._rotate_board_from_left(bcopy, move)
        return bcopy

    def spawn_tile(self):
        """
        Place a new tile in an empty spot:
         - 90% chance of a '2', 10% chance of a '4'
         - If no empty spots remain, do nothing
        """
        empties = []
        for r in range(SIZE):
            for c in range(SIZE):
                if self.board.grid[r][c] == 0:
                    empties.append((r,c))
        if not empties:
            # no empty => check if terminal
            self.board.terminal = self.check_game_over()
            return

        r, c = random.choice(empties)
        tile = 2 if random.random() < 0.9 else 4
        self.board.grid[r][c] = tile
        # Re-check if game is over after placing a tile
        self.board.terminal = self.check_game_over()

    def getAllowedActions(self):
        """
        Return a list of all moves (0..3) that are legal from the current state.
        """
        allowed = []
        for mv in [UP, RIGHT, DOWN, LEFT]:
            if self.isMoveLegal(mv):
                allowed.append(mv)
        return allowed

    def check_game_over(self) -> bool:
        """
        Check if the game is over: 
         - If there's any empty cell, it's not over.
         - If any move merges or slides, it's not over.
         - Otherwise, it's terminal.
        """
        # Check for empty space
        for r in range(SIZE):
            for c in range(SIZE):
                if self.board.grid[r][c] == 0:
                    return False

        # If no empty spaces remain, see if any move is possible
        for mv in [UP, RIGHT, DOWN, LEFT]:
            if self.isMoveLegal(mv):
                return False
        return True

    ########################################################################
    # Internal Helpers: rotating/transposing and sliding
    ########################################################################

    def _rotate_board_to_left(self, b: Board, move: int):
        """
        Transform the board 'b' so that applying 'move' (0..3) 
        corresponds to a left move. The typical strategy:
          - If move=UP   (0), rotate left 90 deg
          - If move=RIGHT(1), rotate 180 deg
          - If move=DOWN (2), rotate right 90 deg
          - If move=LEFT (3), do nothing
        We'll do it in-place for efficiency.
        """
        if move == UP:
            # Rotate 90 left
            self._rotate_90_left(b)
        elif move == RIGHT:
            # Rotate 180
            self._rotate_180(b)
        elif move == DOWN:
            # Rotate 90 right
            self._rotate_90_right(b)
        else:
            # LEFT => do nothing
            pass

    def _rotate_board_from_left(self, b: Board, move: int):
        """
        Undo the transformation from _rotate_board_to_left.
          - If we rotated 90 left, we rotate 90 right
          - If we rotated 180, we rotate 180
          - If we rotated 90 right, we rotate 90 left
          - If we did nothing, we do nothing
        """
        if move == UP:
            self._rotate_90_right(b)
        elif move == RIGHT:
            self._rotate_180(b)
        elif move == DOWN:
            self._rotate_90_left(b)
        else:
            pass

    def _slide_left_and_merge(self, b: Board) -> int:
        """
        Slide tiles to the left, merging them. Return the sum of merged tile values
        as 'merge_reward'. For example, if we merge two '4's into '8',
        we add '8' to the reward. If multiple merges happen, sum them up.
        """
        merge_reward = 0
        for row_idx in range(SIZE):
            row = b.grid[row_idx]

            # 1. filter out zeros:
            filtered = [x for x in row if x != 0]

            # 2. merge from left to right
            merged_row = []
            skip = False
            for i in range(len(filtered)):
                if skip:
                    skip = False
                    continue
                if i+1 < len(filtered) and filtered[i] == filtered[i+1]:
                    # Merge them
                    new_val = filtered[i] * 2
                    merged_row.append(new_val)
                    merge_reward += new_val  # reward for this merge
                    skip = True
                else:
                    merged_row.append(filtered[i])

            # 3. pad with zeros on the right
            while len(merged_row) < SIZE:
                merged_row.append(0)

            b.grid[row_idx] = merged_row
        return merge_reward

    def _rotate_90_left(self, b: Board):
        """
        Rotate board 90 deg left in-place.
        """
        N = SIZE
        grid = b.grid
        # Transpose + flip horizontally => 90 deg left
        # But let's do it a direct way: new[r][c] = old[c][N-1-r]
        new_grid = [[0]*N for _ in range(N)]
        for r in range(N):
            for c in range(N):
                new_grid[r][c] = grid[c][N-1-r]
        b.grid = new_grid

    def _rotate_90_right(self, b: Board):
        """
        Rotate board 90 deg right in-place.
        """
        N = SIZE
        grid = b.grid
        # new[r][c] = old[N-1-c][r]
        new_grid = [[0]*N for _ in range(N)]
        for r in range(N):
            for c in range(N):
                new_grid[r][c] = grid[N-1-c][r]
        b.grid = new_grid

    def _rotate_180(self, b: Board):
        """
        Rotate board 180 deg in-place.
        (Equivalent to rotate left 90 + rotate left 90, or flipping both axes.)
        """
        N = SIZE
        grid = b.grid
        new_grid = [[0]*N for _ in range(N)]
        for r in range(N):
            for c in range(N):
                new_grid[r][c] = grid[N-1-r][N-1-c]
        b.grid = new_grid
        
    def printBoard(self):
        print("----------------")
        for r in range(SIZE):
            for c in range(SIZE):
                print(self.board.grid[r][c], end="\t")
            print()
        print("----------------")

