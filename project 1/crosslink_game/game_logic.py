# game_logic.py
from board import Board
from path_finder import find_shortest_path, find_path_with_turn_cost

class Game:
    def __init__(self, board_size=8):
        self.board = Board(board_size)
        self.pieces = []  # 格式：[{"id": 1, "color": "#ff0000", "positions": [(0,0), (1,1)]}, ...]
        self.paths = []   # 存储找到的路径
    
    def add_piece_pair(self, color, pos1, pos2):
        """添加一对棋子"""
        # 验证位置是否可用
        if not self.is_position_valid(pos1) or not self.is_position_valid(pos2):
            return False, "位置超出棋盘范围"
        
        # 检查位置是否相同
        if pos1 == pos2:
            return False, "两个棋子不能在同一位置"
        
        # 检查是否已被占用
        for piece in self.pieces:
            for pos in piece["positions"]:
                if pos == pos1 or pos == pos2:
                    return False, "位置已被其他棋子占用"
        
        piece_id = len(self.pieces) + 1
        self.pieces.append({
            "id": piece_id,
            "color": color,
            "positions": [pos1, pos2]
        })
        return True, piece_id
    
    def remove_piece_pair(self, piece_id):
        """删除指定ID的棋子对"""
        for i, piece in enumerate(self.pieces):
            if piece["id"] == piece_id:
                self.pieces.pop(i)
                # 删除相关路径
                self.clear_paths()
                return True
        return False
    
    def clear_pieces(self):
        """清除所有棋子"""
        self.pieces = []
        self.clear_paths()
    
    def clear_paths(self):
        """清除所有路径"""
        self.paths = []
    
    def is_position_valid(self, pos):
        """检查位置是否在棋盘范围内"""
        row, col = pos
        return 0 <= row < self.board.size and 0 <= col < self.board.size
    
    def find_all_paths(self, consider_turns=False):
        """为所有棋子对找到最短路径"""
        self.clear_paths()
        results = [] # 记录结果
        all_connected = True
        
        for i, piece in enumerate(self.pieces):
            start_pos = piece["positions"][0]
            end_pos = piece["positions"][1]
            
            # 收集所有被占用的单元格
            occupied_cells = set()
            
            # 添加所有棋子位置
            for p in self.pieces:
                for pos in p["positions"]:
                    occupied_cells.add(pos)
            
            # 添加已有路径占用的单元格
            for path in self.paths:
                for pos in path:
                    # 跳过路径的起点和终点
                    if pos == path[0] or pos == path[-1]:
                        continue
                    occupied_cells.add(pos)
            
            # 查找路径
            if consider_turns:
                path = find_path_with_turn_cost(start_pos, end_pos, occupied_cells, self.board.size)
            else:
                path = find_shortest_path(start_pos, end_pos, occupied_cells, self.board.size)
            
            # 处理结果
            if path:
                self.paths.append(path)
                
                if consider_turns:
                    # 计算路径长度和转向次数
                    path_length = len(path)
                    turns = 0
                    
                    for j in range(1, len(path) - 1):
                        prev = path[j-1]
                        curr = path[j]
                        next_pos = path[j+1]
                        
                        prev_dir = (curr[0] - prev[0], curr[1] - prev[1])
                        next_dir = (next_pos[0] - curr[0], next_pos[1] - curr[1])
                        
                        if prev_dir != next_dir:
                            turns += 1
                    
                    total_cost = path_length + turns * 2
                    results.append({
                        "piece_id": piece["id"],
                        "color": piece["color"],
                        "path_length": path_length,
                        "turns": turns,
                        "total_cost": total_cost,
                        "success": True
                    })
                else:
                    results.append({
                        "piece_id": piece["id"],
                        "color": piece["color"],
                        "path_length": len(path),
                        "success": True
                    })
            else:
                all_connected = False
                results.append({
                    "piece_id": piece["id"],
                    "color": piece["color"],
                    "success": False
                })
        
        return {
            "all_connected": all_connected,
            "results": results
        }
    
    def resize_board(self, new_size):
        """调整棋盘大小"""
        if new_size < 3: # 棋盘大小小于3时操作空间太小
            return False, "棋盘大小不能小于3"
        
        # 检查是否有超出范围的棋子
        for piece in self.pieces:
            for row, col in piece["positions"]:
                if row >= new_size or col >= new_size:
                    return False, "有棋子将超出新棋盘范围"
        
        self.board = Board(new_size)
        self.clear_paths()  # 调整棋盘后需要重新计算路径
        return True, "棋盘大小已调整"