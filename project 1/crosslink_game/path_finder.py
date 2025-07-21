# path_finder.py
import heapq

def manhattan_distance(pos1, pos2):
    """计算曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def find_shortest_path(start, end, occupied_cells, board_size):
    """使用A*算法找到最短路径（不考虑转向代价）"""
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
    
    open_set = []
    heapq.heappush(open_set, (manhattan_distance(start, end), 0, manhattan_distance(start, end), start, [start]))
    
    # 已访问节点集合
    closed_set = set()
    
    while open_set:
        _, g, _, pos, path = heapq.heappop(open_set)
        
        # 到达终点
        if pos == end:
            return path
        
        # 将当前节点标记为已访问
        if pos in closed_set:
            continue
        closed_set.add(pos)
        
        # 探索相邻节点
        for dr, dc in directions:
            next_row, next_col = pos[0] + dr, pos[1] + dc
            next_pos = (next_row, next_col)
            
            # 检查是否在棋盘内
            if not (0 <= next_row < board_size and 0 <= next_col < board_size):
                continue
            
            # 检查是否是障碍物(已占用的路径)，始终可以经过终点
            if next_pos in occupied_cells and next_pos != end:
                continue
            
            # 检查是否已访问
            if next_pos in closed_set:
                continue
            
            # 计算代价
            new_g = g + 1
            h = manhattan_distance(next_pos, end)
            f = new_g + h
            
            # 添加到开放集合
            heapq.heappush(open_set, (f, new_g, h, next_pos, path + [next_pos]))
    
    # 没有找到路径
    return None

def find_shortest_path_with_cutting(start, end, occupied_cells, board_size):
    """使用A*算法找到最短路径（不考虑转向代价），并加入剪枝"""
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
    open_set = []
    visited = {}  # 记录每个状态的最小代价
    best_cost = float('inf')  # 当前已知的最优代价

    # 初始化起点
    heapq.heappush(open_set, (manhattan_distance(start, end), 0, start, [start]))

    while open_set:
        f, g, pos, path = heapq.heappop(open_set)

        # 剪枝：如果当前总代价超过已知最优代价，跳过
        if f >= best_cost:
            continue

        # 如果到达终点，更新最优代价并返回路径
        if pos == end:
            best_cost = g
            return path

        # 检查是否已经访问过当前状态
        if pos in visited and visited[pos] <= g:
            continue
        visited[pos] = g

        # 扩展相邻节点
        for dr, dc in directions:
            next_row, next_col = pos[0] + dr, pos[1] + dc
            next_pos = (next_row, next_col)

            # 检查是否在棋盘范围内
            if not (0 <= next_row < board_size and 0 <= next_col < board_size):
                continue

            # 检查是否是障碍物（已占用的路径），始终可以经过终点
            if next_pos in occupied_cells and next_pos != end:
                continue

            # 计算代价
            new_g = g + 1
            h = manhattan_distance(next_pos, end)
            f = new_g + h

            # 剪枝：如果当前总代价超过已知最优代价，跳过
            if f >= best_cost:
                continue

            # 添加到开放集合
            heapq.heappush(open_set, (f, new_g, next_pos, path + [next_pos]))

    # 如果没有找到路径，返回None
    return None

def find_path_with_turn_cost(start, end, occupied_cells, board_size):
    """使用A*算法找到最短路径（考虑转向代价）"""
    directions = [
        (-1, 0, 'U'),  
        (1, 0, 'D'),   
        (0, -1, 'L'),  
        (0, 1, 'R')    
    ]
    

    open_set = []
    # 添加4个起始状态，每个方向一个
    for dr, dc, direction in directions:
        heapq.heappush(open_set, (
            manhattan_distance(start, end), 
            0, 
            manhattan_distance(start, end), 
            start, 
            [start], 
            direction
        ))
    
    # 已访问节点集合 (需要考虑方向)
    closed_set = set()
    
    while open_set:
        _, g, _, pos, path, last_dir = heapq.heappop(open_set)
        
        # 如果到达终点
        if pos == end:
            return path
        
        # 将当前节点和方向标记为已访问
        state = (pos, last_dir)
        if state in closed_set:
            continue
        closed_set.add(state)
        
        # 探索相邻节点
        for dr, dc, direction in directions:
            next_row, next_col = pos[0] + dr, pos[1] + dc
            next_pos = (next_row, next_col)
            
            # 检查是否在棋盘内
            if not (0 <= next_row < board_size and 0 <= next_col < board_size):
                continue
            
            # 检查是否是障碍物(已占用的路径)，始终可以经过终点
            if next_pos in occupied_cells and next_pos != end and next_pos != start:
                continue
            
            # 计算代价
            new_g = g + 1
            
            # 如果方向改变，增加转向代价
            if direction != last_dir:
                new_g += 2
            
            h = manhattan_distance(next_pos, end)
            f = new_g + h
            
            # 检查是否已访问
            new_state = (next_pos, direction)
            if new_state in closed_set:
                continue
            
            # 添加到开放集合
            heapq.heappush(open_set, (f, new_g, h, next_pos, path + [next_pos], direction))
    
    # 没有找到路径
    return None

def find_path_with_turn_cost_with_cutting(start, end, occupied_cells, board_size):
    """使用A*算法找到最短路径（考虑转向代价），并加入剪枝"""
    directions = [
        (-1, 0, 'U'),  
        (1, 0, 'D'),   
        (0, -1, 'L'),  
        (0, 1, 'R')    
    ]
    open_set = []
    visited = {}  # 记录每个状态的最小代价
    best_cost = float('inf')  # 当前已知的最优代价

    # 初始化起点
    for dr, dc, direction in directions:
        heapq.heappush(open_set, (
            manhattan_distance(start, end), 
            0, 
            start, 
            [start], 
            direction
        ))

    while open_set:
        f, g, pos, path, last_dir = heapq.heappop(open_set)

        # 剪枝：如果当前总代价超过已知最优代价，跳过
        if f >= best_cost:
            continue

        # 如果到达终点，更新最优代价并返回路径
        if pos == end:
            best_cost = g
            return path

        # 检查是否已经访问过当前状态
        state = (pos, last_dir)
        if state in visited and visited[state] <= g:
            continue
        visited[state] = g

        # 扩展相邻节点
        for dr, dc, direction in directions:
            next_row, next_col = pos[0] + dr, pos[1] + dc
            next_pos = (next_row, next_col)

            # 检查是否在棋盘范围内
            if not (0 <= next_row < board_size and 0 <= next_col < board_size):
                continue

            # 检查是否是障碍物（已占用的路径），始终可以经过终点
            if next_pos in occupied_cells and next_pos != end and next_pos != start:
                continue

            # 计算代价
            new_g = g + 1
            if direction != last_dir:  # 转向代价
                new_g += 2
            h = manhattan_distance(next_pos, end)
            f = new_g + h

            # 剪枝：如果当前总代价超过已知最优代价，跳过
            if f >= best_cost:
                continue

            # 添加到开放集合
            heapq.heappush(open_set, (f, new_g, next_pos, path + [next_pos], direction))

    # 如果没有找到路径，返回None
    return None