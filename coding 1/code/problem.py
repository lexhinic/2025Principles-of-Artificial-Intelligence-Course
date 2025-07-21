import numpy as np
from copy import deepcopy
from utils import *
import time
import math

def find_loc(state, num):
    loc = np.where(np.array(state) == num)
    return (loc[0][0], loc[1][0])

def loc_of_goal_state(state):
    goal_loc = {}
    for i in [0, 2, 3, 4, 5, 6]:
        goal_loc[i] = find_loc(state, i)
    return goal_loc

goal_loc = loc_of_goal_state([[1, 1, 1, 1], 
                             [2, 1, 1, 1], 
                             [3, 5, 1, 1], 
                             [4, 6, 1, 0]])

goal_loc = {0: (3, 3), 2: (1, 0), 3: (2, 0), 4: (3, 0), 5: (2, 1), 6: (3, 1)}

def h_function_null(state, goal_state):
    return 0

def h_function_method2(state, goal_state):
    dis = 0
    for i in [2, 3, 4, 6]:
        dis += abs(find_loc(state, i)[0] - goal_loc[i][0]) + abs(find_loc(state, i)[1] - goal_loc[i][1])
    if find_loc(state, 5)[0] != goal_loc[5][0] or find_loc(state, 5)[1] != goal_loc[5][1]:
        dis += 1
    return dis

def h_function_method3(state, goal_state):
    dis = 0
    for i in [2, 3, 4, 5, 6]:
        if find_loc(state, i)[0] != goal_loc[i][0] or find_loc(state, i)[1] != goal_loc[i][1]:
            dis += 1
    return dis

def h_function_method4(state, goal_state):
    dis = 0
    for i in [2, 3, 4, 6]:
        current_loc = find_loc(state, i) 
        dis += math.sqrt((current_loc[0] - goal_loc[i][0]) ** 2 + (current_loc[1] - goal_loc[i][1]) ** 2)
    
    if find_loc(state, 5)[0] != goal_loc[5][0] or find_loc(state, 5)[1] != goal_loc[5][1]:
        dis += 1
    
    return dis


class Node(object):  # Represents a node in a search tree
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def child_node(self, problem, action):
        next_state = problem.move(self.state, action)
        next_node = Node(next_state, self, action,
                         problem.g(self.depth, self.state, action, next_state) + 
                         problem.h(next_state, problem.goal_state.state),)
        return next_node

    def path(self):
        """
        Returns list of nodes from this node to the root node
        """
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __repr__(self):
        return f"#########\n#{self.state[0][0]} {self.state[0][1]} {self.state[0][2]} {self.state[0][3]}#\n#{self.state[1][0]} {self.state[1][1]} {self.state[1][2]} {self.state[1][3]}#\n#{self.state[2][0]} {self.state[2][1]} {self.state[2][2]} {self.state[2][3]}#\n#{self.state[3][0]} {self.state[3][1]} {self.state[3][2]} {self.state[3][3]}#\n#########"


    def __lt__(self, other):
        return self.path_cost < other.path_cost

    def __eq__(self, other):
        return self.state == other.state


class Problem(object):
    def __init__(self, init_state=None, goal_state=None, h_function=None):
        self.init_state = Node(init_state)
        self.goal_state = Node(goal_state)
        self.h = h_function

    def actions(self, state):
        """
        Given the current state, return valid actions.
        :param state:
        :return: valid actions
        """
        pass

    def move(self, state, action):
        pass

    def is_goal(self, state):
        pass

    def g(self, cost, from_state, action, to_state):
        return cost + 1

    def solution(self, goal):
        """
        Returns actions from this node to the root node
        """
        if goal.state is None:
            return None
        return [node.action for node in goal.path()[1:]]

    def expand(self, node):  # Returns a list of child nodes
        return [node.child_node(self, action) for action in self.actions(node.state)]
    

class GridsProblem(Problem):
    def __init__(self,
                 n,
                 init_state=[[1, 3, 1, 1], 
                             [1, 1, 0, 1], 
                             [1, 2, 1, 1], 
                             [4, 1, 5, 6]],
#                 init_state=[[3, 1, 0, 1], 
#                             [1, 1, 1, 1], 
#                             [1, 1, 1, 5], 
#                             [4, 1, 2, 6]],
                 goal_state=[[1, 1, 1, 1], 
                             [2, 1, 1, 1], 
                             [3, 5, 1, 1], 
                             [4, 6, 1, 0]],
                 h_function=h_function_null):
        super().__init__(init_state, goal_state, h_function)
        self.n = n
        
    def valid_state(self, state):
        nezha_head_row, nezha_head_col = find_loc(state, 3)
        nezha_body_row, nezha_body_col = find_loc(state, 4)
        dis = abs(nezha_head_row - nezha_body_row) + abs(nezha_head_col - nezha_body_col)
        return dis <= 4
    
    def is_valid(self, loc):
        return -1 < loc[0] < self.n and -1 < loc[1] < self.n

    def actions(self, state):
        aobing_row, aobing_col = find_loc(state, 5)
        empty_row, empty_col = np.where(np.array(state) == 0)[0][0], np.where(np.array(state) == 0)[1][0]
        candidates = [[empty_row-1, empty_col], [empty_row+1, empty_col],
                      [empty_row, empty_col-1], [empty_row, empty_col+1], [aobing_row, aobing_col]]
        valid_candidates = [item for item in candidates if (self.is_valid(item) and self.valid_state(self.move(state, item)))]
        return valid_candidates

    def move(self, state, action):
        empty_row, empty_col = np.where(np.array(state) == 0)[0][0], np.where(np.array(state) == 0)[1][0]
        new_state = deepcopy(state)
        new_state[empty_row][empty_col] = state[action[0]][action[1]]
        new_state[action[0]][action[1]] = 0
        return new_state

    def is_goal(self, state):
        return state == self.goal_state.state

    def g(self, cost, from_state, action, to_state):
        return cost + 1
    
    
def search_with_info(problem: GridsProblem):
    print("有信息搜索。")
    start_time = time.time()
    q = PriorityQueue(problem.init_state)
    explored = {}  
    nodes = 0
    while not q.empty():
        node = q.pop()
        nodes += 1
        state_key = tuple(tuple(row) for row in node.state)
        # 剪枝操作：如果该状态已处理且当前路径成本不优于已知值，跳过
        if state_key in explored and node.path_cost >= explored[state_key]:
            continue
        # 更新当前状态的最小成本
        explored[state_key] = node.path_cost
        if problem.is_goal(node.state):
            print("Goal state reached.")
            print(f"Solution: {problem.solution(node)}")
            print(f"访问的节点数：{nodes}")
            print(f"路径长度：{node.depth}")
            print(f"算法运行时间：{time.time() - start_time}")
            return
        for child in problem.expand(node):
            child_state_key = tuple(tuple(row) for row in child.state)
            # 剪枝操作：仅当子节点状态未探索或路径成本更低时加入队列
            if child_state_key not in explored or child.path_cost < explored.get(child_state_key, float('inf')):
                q.push(child)
    print("Goal state not reached.")
    return

def search_without_info(problem: GridsProblem):
    print("无信息搜索")
    start_time = time.time()
    q = Queue()
    q.push(problem.init_state)
    explored = set()
    nodes = 0
    while not q.empty():
        node = q.pop()
        nodes += 1
#        print(node)
        if problem.is_goal(node.state):
            print("Goal state reached.")
            print(f"Solution: {problem.solution(node)}")
            print(f"访问的节点数：{nodes}")
            print(f"路径长度：{node.depth}")
            print(f"算法运行的时间：{time.time() - start_time}")
            return
        explored.add(tuple(tuple(sublist) for sublist in node.state))
        for child in problem.expand(node):
            if tuple(tuple(sublist) for sublist in child.state) not in explored:
                q.push(child)
    print("Goal state not reached.")
    return



if __name__ == "__main__":
    problem = GridsProblem(4, h_function=h_function_null)
    start_time = time.time()
    search_without_info(problem)
    end_time = time.time()

    problem = GridsProblem(4, h_function=h_function_null)
    start_time = time.time()
    search_with_info(problem)
    end_time = time.time()

    problem = GridsProblem(4, h_function=h_function_method2)
    start_time = time.time()
    search_with_info(problem)
    end_time = time.time()

    problem = GridsProblem(4, h_function=h_function_method3)
    start_time = time.time()
    search_with_info(problem)
    end_time = time.time()

    problem = GridsProblem(4, h_function=h_function_method4)
    start_time = time.time()
    search_with_info(problem)
    end_time = time.time()