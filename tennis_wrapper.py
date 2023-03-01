import gym
import random
import numpy as np
import torch

#解决重开一小局的僵直问题(在每一球初始化同时，不断执行Noop操作直至僵直问题不存在)，
#开局摆烂问题(判定我方发球状态、找出非法动作并将奖励设置为-1)
class TennisWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.current_score = [0,0] #小比分
        self.score = [0,0] #大比分
        self.all_score = [0,0] #总比分
        self.server = 0 #发球手
        obs = self.run_reset(obs)
        return obs

    #每个回合reset
    def run_reset(self,old_obs,max_loop=1000):
        self.actual_length = 0
        self.run_length = 0
        for _ in range(max_loop):
            obs = self.env.step(0)[0]
            if not np.all(old_obs == obs):
                break
            old_obs = obs
        return obs #一定要返回新游戏的初始状态
        
    def step(self, action):
        #我方发球的情况下惩罚非法动作
        illegal_action = False
        if self.server == 0 and self.actual_length == 0: #我方发球
            if action in [0] + list(range(2,10)): #非法动作
                illegal_action = True
        #执行相应的动作
        obs, reward, done, info = self.env.step(action)
        self.actual_length = self.actual_length + 1 if not illegal_action else self.actual_length
        self.run_length += 1
        info["run_done"] = False        
        #每回合结束
        if reward != 0:
            info["run_done"] = True
            info["actual_length"] = self.actual_length
            info["run_length"] = self.run_length
            reward += self.actual_length / 50
            run_winner = 0 if reward == 1 else 1
            self.current_score[run_winner] += 1
            self.all_score[run_winner] += 1
            # 每小局胜利条件
            if self.current_score[run_winner] >= 4 and self.current_score[run_winner] - self.current_score[1-run_winner]>=2:      
                self.current_score = [0,0]
                self.score[run_winner] += 1
                self.server = 1 - self.server
            obs = self.run_reset(obs.copy())
        reward = -1 if illegal_action else reward
        return obs, reward, done, info

#添加action_mask解决开局摆烂问题.由于reset返回obs和info两个信息时会对后续的wrapper有影响，这边索性就保留action_mask这个接口来显示调用
class TennisWrapper2(gym.Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.current_score = [0,0] #小比分
        self.score = [0,0] #大比分
        self.all_score = [0,0] #总比分
        self.server = 0 #发球手
        obs = self.run_reset(obs)
        return obs

    @property
    def action_mask(self):
        am = np.array([True]*self.action_space.n)
        if self.server == 0 and self.run_length == 0:
            am[[0] + list(range(2,10))] = False
        return am

    #每个回合reset
    def run_reset(self,old_obs,max_loop=1000):
        self.run_length = 0
        for _ in range(max_loop):
            obs = self.env.step(0)[0]
            if not np.all(old_obs == obs):
                break
            old_obs = obs
        return obs #一定要返回新游戏的初始状态
        
    def step(self, action):
        #判断action是否illegel
        assert self.action_mask[action], "not a legal action"
        #执行相应的动作
        obs, reward, done, info = self.env.step(action)
        self.run_length += 1
        info["run_done"] = False        
        #每回合结束
        if reward != 0:
            info["run_done"] = True
            info["run_length"] = self.run_length
            # reward += info["run_length"] / 50
            run_winner = 0 if reward == 1 else 1
            self.current_score[run_winner] += 1
            self.all_score[run_winner] += 1
            # 每小局胜利条件
            if self.current_score[run_winner] >= 4 and self.current_score[run_winner] - self.current_score[1-run_winner]>=2:      
                self.current_score = [0,0]
                self.score[run_winner] += 1
                self.server = 1 - self.server #更换发球者
            obs = self.run_reset(obs.copy())
        info["action_mask"] = self.action_mask
        return obs, reward, done, info


if __name__ == "__main__":
    env = gym.make("Tennis-v4")
    env = TennisWrapper2(env)
    envs = gym.vector.SyncVectorEnv([lambda: env for _ in range(4)])
    print(envs.single_action_space.n)
    print(envs.reset().shape)
    print([env.action_mask for env in envs.envs])

    action_masks = torch.Tensor([env.action_mask for env in envs.envs]).reshape(4, 18)
    print(action_masks.shape)

    # print(envs.action_mask)
    
    # env = gym.wrappers.ResizeObservation(env, (84, 84))
    # env = gym.wrappers.GrayScaleObservation(env)
    # env = gym.wrappers.FrameStack(env, 4)
    # for i in range(200):
    #     state = env.reset()
    #     action_mask = env.action_mask
    #     done = False
    #     episode_reward,episode_length = 0,0
    #     # start = time.time()
    #     while not done:
    #         env.render()
    #         action = random.choice(np.where(action_mask)[0])
    #         print(f"steps:{episode_length},action:{action}")
    #         next_state, reward, done, info = env.step(action)
    #         action_mask = env.action_mask
    #         print(f"steps:{episode_length},server:{env.server},current_score:{env.current_score},reward:{reward},done:{done},info:{info}")
    #         episode_reward += reward
    #         episode_length += 1
    #         state = next_state
