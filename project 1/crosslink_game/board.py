# board.py
class Board:
    def __init__(self, size=8):
        self.size = size
        self.grid = [[None for _ in range(size)] for _ in range(size)]
    
    def is_in_bounds(self, row, col):
        """检查坐标是否在棋盘范围内"""
        return 0 <= row < self.size and 0 <= col < self.size