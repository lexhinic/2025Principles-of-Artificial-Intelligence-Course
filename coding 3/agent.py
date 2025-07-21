import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        self.actions = actions  # 动作空间
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # 折扣因子
        self.epsilon = e_greedy  # 探索率
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.step_counts = []  
        
    def check_state_exist(self, state):
        if state != 'terminal' and str(state) not in self.q_table.index:
            new_row = pd.DataFrame(
                [[0] * len(self.actions)],  
                index=[str(state)],
                columns=self.q_table.columns
            )
            self.q_table = pd.concat([self.q_table, new_row])
            
    def choose_action(self, observation):
        self.check_state_exist(observation)
        
        # ε-greedy策略
        if np.random.uniform() < self.epsilon:
            # 随机选择动作（探索）
            action = np.random.choice(self.actions)
        else:
            # 选择Q值最大的动作（利用）
            state_action = self.q_table.loc[str(observation), :]
            # 处理相同Q值的情况
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        return action
    
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        
        q_predict = self.q_table.loc[str(s), a]
        
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[str(s_), :].max()
        else:
            q_target = r
        
        # 更新Q值
        self.q_table.loc[str(s), a] += self.lr * (q_target - q_predict)
    
    def plot_learning_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.step_counts)), self.step_counts)
        plt.xlabel('Episodes')  
        plt.ylabel('Steps')    
        plt.title('Learning Curve') 
        plt.savefig('learning_curve.png')
        plt.close()

# Q(St, At) ← Q(St, At) + α[Rt+1 + γRt+2 + γ² max_a Q(St+2, a) - Q(St, At)]       
class TwoStepQLearningAgent(QLearningAgent):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        super().__init__(actions, learning_rate, reward_decay, e_greedy)
        # 存储状态、动作和奖励
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
    def store_transition(self, s, a, r):
        self.state_history.append(s)
        self.action_history.append(a)
        self.reward_history.append(r)
        
        if len(self.state_history) > 3:
            self.state_history.pop(0)
            self.action_history.pop(0)
            self.reward_history.pop(0)
        
    def learn(self):
        if len(self.state_history) < 3:
            # 如果历史不足2步，无法进行2-step更新
            return
        
        # 获取t时刻的状态和动作
        s_t = self.state_history[-3]
        a_t = self.action_history[-3]
        
        # 获取t+1和t+2时刻的奖励
        r_t1 = self.reward_history[-2]
        r_t2 = self.reward_history[-1]
        
        # 获取t+2时刻的状态
        s_t2 = self.state_history[-1]
        
        self.check_state_exist(s_t)
        self.check_state_exist(s_t2)
        
        # 当前Q值预测
        q_predict = self.q_table.loc[str(s_t), a_t]
        
        if s_t2 != 'terminal':
            q_target = r_t1 + self.gamma * r_t2 + self.gamma**2 * self.q_table.loc[str(s_t2), :].max()
        else:
            q_target = r_t1 + self.gamma * r_t2
        
        # 更新Q值
        self.q_table.loc[str(s_t), a_t] += self.lr * (q_target - q_predict)
        
        # 保持历史最多存储3个时间步
        if len(self.state_history) > 3:
            self.state_history.pop(0)
            self.action_history.pop(0)
            self.reward_history.pop(0)