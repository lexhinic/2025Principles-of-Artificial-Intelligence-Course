# path_finder.py
import heapq
import time

def manhattan_distance(pos1, pos2):
    """计算曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def find_shortest_path(start, end, occupied_cells, board_size):
    """使用A*算法找到最短路径（不考虑转向代价）"""
    print("\n==== 不考虑转向代价的A*搜索 ====")
    start_time = time.time()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
    
    # 优先队列，用于A*搜索
    open_set = []
    # 项目格式：(f, g, h, pos, path)，f是估计总代价，g是从起点到当前的代价，h是启发式函数值
    heapq.heappush(open_set, (manhattan_distance(start, end), 0, manhattan_distance(start, end), start, [start]))
    
    # 已访问节点集合
    closed_set = set()
    
    # 性能统计
    nodes_explored = 0
    max_queue_size = 1
    
    while open_set:
        max_queue_size = max(max_queue_size, len(open_set))
        _, g, _, pos, path = heapq.heappop(open_set)
        nodes_explored += 1
        
        # 如果到达终点
        if pos == end:
            elapsed_time = time.time() - start_time
            print(f"路径已找到! 起点: {start}, 终点: {end}")
            print(f"路径长度: {len(path)}")
            print(f"搜索统计:")
            print(f"- 探索节点数: {nodes_explored}")
            print(f"- 最大队列大小: {max_queue_size}")
            print(f"- 搜索时间: {elapsed_time:.6f} 秒")
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
    elapsed_time = time.time() - start_time
    print(f"未找到路径! 起点: {start}, 终点: {end}")
    print(f"搜索统计:")
    print(f"- 探索节点数: {nodes_explored}")
    print(f"- 最大队列大小: {max_queue_size}")
    print(f"- 搜索时间: {elapsed_time:.6f} 秒")
    return None

def find_shortest_path_with_cutting(start, end, occupied_cells, board_size):
    """使用A*算法找到最短路径（不考虑转向代价），并加入剪枝"""
    print("\n==== 带剪枝的A*搜索 (不考虑转向代价) ====")
    start_time = time.time()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
    open_set = []
    visited = {}  # 记录每个状态的最小代价
    best_cost = float('inf')  # 当前已知的最优代价

    # 初始化起点
    heapq.heappush(open_set, (manhattan_distance(start, end), 0, start, [start]))

    # 性能统计
    nodes_explored = 0
    nodes_pruned = 0
    max_queue_size = 1

    while open_set:
        max_queue_size = max(max_queue_size, len(open_set))
        f, g, pos, path = heapq.heappop(open_set)
        nodes_explored += 1

        # 剪枝：如果当前总代价超过已知最优代价，跳过
        if f >= best_cost:
            nodes_pruned += 1
            continue

        # 如果到达终点，更新最优代价并返回路径
        if pos == end:
            best_cost = g
            elapsed_time = time.time() - start_time
            print(f"路径已找到! 起点: {start}, 终点: {end}")
            print(f"路径长度: {len(path)}")
            print(f"搜索统计:")
            print(f"- 探索节点数: {nodes_explored}")
            print(f"- 剪枝节点数: {nodes_pruned}")
            print(f"- 最大队列大小: {max_queue_size}")
            print(f"- 搜索时间: {elapsed_time:.6f} 秒")
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
                nodes_pruned += 1
                continue

            # 添加到开放集合
            heapq.heappush(open_set, (f, new_g, next_pos, path + [next_pos]))

    # 如果没有找到路径，返回None
    elapsed_time = time.time() - start_time
    print(f"未找到路径! 起点: {start}, 终点: {end}")
    print(f"搜索统计:")
    print(f"- 探索节点数: {nodes_explored}")
    print(f"- 剪枝节点数: {nodes_pruned}")
    print(f"- 最大队列大小: {max_queue_size}")
    print(f"- 搜索时间: {elapsed_time:.6f} 秒")
    return None

def find_path_with_turn_cost(start, end, occupied_cells, board_size):
    """使用A*算法找到最短路径（考虑转向代价）"""
    print("\n==== 考虑转向代价的A*搜索 ====")
    start_time = time.time()
    directions = [(0, -1, 'L'), (0, 1, 'R'), (-1, 0, 'U'), (1, 0, 'D')]  # 左右上下
    
    # 优先队列，用于A*搜索
    open_set = []
    # 添加4个起始状态，每个方向一个
    for dr, dc, direction in directions:
        # 项目格式：(f, g, h, pos, path, last_direction)
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
    
    # 性能统计
    nodes_explored = 0
    max_queue_size = len(directions)  # 初始有4个方向
    turns_count = 0
    
    while open_set:
        max_queue_size = max(max_queue_size, len(open_set))
        _, g, _, pos, path, last_dir = heapq.heappop(open_set)
        nodes_explored += 1
        
        # 如果到达终点
        if pos == end:
            elapsed_time = time.time() - start_time
            print(f"路径已找到! 起点: {start}, 终点: {end}")
            print(f"路径长度: {len(path)}")
            print(f"转向次数: {turns_count}")
            print(f"总代价(长度 + 2*转向): {g}")
            print(f"搜索统计:")
            print(f"- 探索节点数: {nodes_explored}")
            print(f"- 最大队列大小: {max_queue_size}")
            print(f"- 搜索时间: {elapsed_time:.6f} 秒")
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
            turn_happened = False
            if direction != last_dir:
                new_g += 2
                turn_happened = True
            
            h = manhattan_distance(next_pos, end)
            f = new_g + h
            
            # 检查是否已访问
            new_state = (next_pos, direction)
            if new_state in closed_set:
                continue
            
            # 添加到开放集合
            if turn_happened:
                turns_count += 1
            heapq.heappush(open_set, (f, new_g, h, next_pos, path + [next_pos], direction))
    
    # 没有找到路径
    elapsed_time = time.time() - start_time
    print(f"未找到路径! 起点: {start}, 终点: {end}")
    print(f"搜索统计:")
    print(f"- 探索节点数: {nodes_explored}")
    print(f"- 最大队列大小: {max_queue_size}")
    print(f"- 搜索时间: {elapsed_time:.6f} 秒")
    return None

def find_path_with_turn_cost_with_cutting(start, end, occupied_cells, board_size):
    """使用A*算法找到最短路径（考虑转向代价），并加入剪枝"""
    print("\n==== 带剪枝的A*搜索 (考虑转向代价) ====")
    start_time = time.time()
    directions = [
        (-1, 0, 'U'),  # 上
        (1, 0, 'D'),   # 下
        (0, -1, 'L'),  # 左
        (0, 1, 'R')    # 右
    ]
    open_set = []
    visited = {}  # 记录每个状态的最小代价
    best_cost = float('inf')  # 当前已知的最优代价

    # 性能统计
    nodes_explored = 0
    nodes_pruned = 0
    max_queue_size = len(directions)  # 初始有4个方向
    turns_count = 0

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
        max_queue_size = max(max_queue_size, len(open_set))
        f, g, pos, path, last_dir = heapq.heappop(open_set)
        nodes_explored += 1

        # 剪枝：如果当前总代价超过已知最优代价，跳过
        if f >= best_cost:
            nodes_pruned += 1
            continue

        # 如果到达终点，更新最优代价并返回路径
        if pos == end:
            best_cost = g
            elapsed_time = time.time() - start_time
            print(f"路径已找到! 起点: {start}, 终点: {end}")
            print(f"路径长度: {len(path)}")
            print(f"转向次数: {turns_count}")
            print(f"总代价(长度 + 2*转向): {g}")
            print(f"搜索统计:")
            print(f"- 探索节点数: {nodes_explored}")
            print(f"- 剪枝节点数: {nodes_pruned}")
            print(f"- 最大队列大小: {max_queue_size}")
            print(f"- 搜索时间: {elapsed_time:.6f} 秒")
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
            turn_happened = False
            if direction != last_dir:  # 转向代价
                new_g += 2
                turn_happened = True
            h = manhattan_distance(next_pos, end)
            f = new_g + h

            # 剪枝：如果当前总代价超过已知最优代价，跳过
            if f >= best_cost:
                nodes_pruned += 1
                continue

            # 添加到开放集合
            if turn_happened:
                turns_count += 1
            heapq.heappush(open_set, (f, new_g, next_pos, path + [next_pos], direction))

    # 如果没有找到路径，返回None
    elapsed_time = time.time() - start_time
    print(f"未找到路径! 起点: {start}, 终点: {end}")
    print(f"搜索统计:")
    print(f"- 探索节点数: {nodes_explored}")
    print(f"- 剪枝节点数: {nodes_pruned}")
    print(f"- 最大队列大小: {max_queue_size}")
    print(f"- 搜索时间: {elapsed_time:.6f} 秒")
    return None