from maze_env import Maze
from agent import QLearningAgent, TwoStepQLearningAgent
import time
import numpy as np
import random

# 设定随机种子
SEED = 42  
np.random.seed(SEED)
random.seed(SEED)

def train_agent():
    EPISODES = 100  # 训练回合数
    RENDER = False  # 是否在训练过程中渲染环境
    
    # 初始化智能体
    agent = QLearningAgent(
        actions=list(range(env.n_actions)),
        learning_rate=0.1,
        reward_decay=0.5,
        e_greedy=0.01
    )
    
    env.withdraw() 
    
    # 训练
    for episode in range(EPISODES):
        # 初始化环境
        observation = env.reset()
        step_count = 0
        
        while True:
            if RENDER:
                env.render()
            
            # 智能体选择动作
            action = agent.choose_action(str(observation))
            
            # 环境执行动作，获得下一个状态和奖励
            observation_, reward, done = env.step(action)
            
            # 智能体学习
            agent.learn(str(observation), action, reward, observation_)
            
            # 更新状态
            observation = observation_
            step_count += 1
            
            # 回合结束，记录步数并退出循环
            if done:
                agent.step_counts.append(step_count)
                print(f'回合 {episode+1}: {step_count} 步')
                break
    
    # 学习曲线
    agent.plot_learning_curve()
    print("训练完成，绘制学习曲线...")
    
    # 最终的Q表
    print('Q-table:')
    print(agent.q_table)
    
    # 最终学习到的策略
    env.deiconify() 
    print("演示最优策略...")
    RENDER = True
    observation = env.reset()
    
    while True:
        if RENDER:
            env.render()
        time.sleep(0.5)  #
        # 使用最优策略
        state_action = agent.q_table.loc[str(observation), :]
        action = state_action.idxmax()
        
        observation_, reward, done = env.step(action)
        observation = observation_
        
        if done:
            break
        
def train_two_step_agent():
    EPISODES = 100 
    RENDER = False  
    
    # 2-step Q-learning智能体
    two_step_agent = TwoStepQLearningAgent(
        actions=list(range(env.n_actions)),
        learning_rate=0.1,
        reward_decay=0.5,
        e_greedy=0.1
    )
    
    env.withdraw()  
    
    # 训练
    for episode in range(EPISODES):
        # 初始化环境
        observation = env.reset()
        step_count = 0
        
        # 清空状态历史
        two_step_agent.state_history = []
        two_step_agent.action_history = []
        two_step_agent.reward_history = []
        
        while True:
            if RENDER:
                env.render()
            
            # 智能体选择动作
            action = two_step_agent.choose_action(str(observation))
            
            # 环境执行动作，获得下一个状态和奖励
            observation_, reward, done = env.step(action)
            
            # 存储转换
            two_step_agent.store_transition(str(observation), action, reward)
            
            # 进行学习
            if len(two_step_agent.state_history) >= 3:
                two_step_agent.learn()
            
            # 更新状态
            observation = observation_
            step_count += 1
            
            # 回合结束，记录步数并退出循环
            if done:
                two_step_agent.store_transition(str(observation_), None, 0)
                if len(two_step_agent.state_history) >= 3:
                    two_step_agent.learn()
                    
                two_step_agent.step_counts.append(step_count)
                print(f'回合 {episode+1}: {step_count} 步 (2-step)')
                break
    
    # 学习曲线
    two_step_agent.plot_learning_curve()
    print("2-step Q-learning训练完成，绘制学习曲线...")
    
    # 最终的Q表
    print('2-step Q-learning Q-table:')
    print(two_step_agent.q_table)
    
    # 最终学习到的策略
    env.deiconify()  
    print("演示2-step Q-learning最优策略...")
    RENDER = True
    observation = env.reset()
    
    while True:
        if RENDER:
            env.render()
        time.sleep(0.5)  
        
        # 使用最优策略
        state_action = two_step_agent.q_table.loc[str(observation), :]
        action = state_action.idxmax()
        
        observation_, reward, done = env.step(action)
        observation = observation_
        
        if done:
            break

if __name__ == "__main__":
    env = Maze()
#    env.after(100, train_agent)
#    env.mainloop()
#    env = Maze()
    env.after(1000, train_two_step_agent)
    env.mainloop()