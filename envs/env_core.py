import numpy as np
import json


class EnvCore(object):
    """
    # 环境中的智能体
    """
    def load_data(self, filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data
    
    def __init__(self):

        # # original version, included in Lognam's testbed as well
        # self.agent_num = 2  # 设置智能体(小飞机)的个数，这里设置为两个 # set the number of agents(aircrafts), here set to two
        # self.obs_dim = 14  # 设置智能体的观测维度 # set the observation dimension of agents
        # self.action_dim = 5  # 设置智能体的动作维度，这里假定为一个五个维度的 # set the action dimension of agents, here set to a five-dimensional



        # Lognam: try to implement scenario
        self.scene_info = None
        self.UAV_targets = None
        self.UAV_og_positions = None

        self.UAV_positions = None

        self.scene_blocks = None

        # self.load_scene_data("scene_data_simple.json")
        # self.load_targets("UAV_targets.json")

        self.scene_info = self.load_data("scene_data_simple.json")
        self.UAV_targets = np.array(self.load_data("UAV_targets.json")['targets'])
        self.UAV_og_positions = np.array(self.load_data("UAV_og_positions.json")['positions'])

        self.scene_blocks = np.array([block['bottomCorner'] + block['size'] + [block['height']] for block in self.scene_info['blocks']])


        # self.agent_num = 2  # 设置智能体(小飞机)的个数，这里设置为两个 # set the number of agents(aircrafts), here set to two
        # self.obs_dim = 14  # 设置智能体的观测维度 # set the observation dimension of agents
        # self.action_dim = 5  # 设置智能体的动作维度，这里假定为一个五个维度的 # set the action dimension of agents, here set to a five-dimensional

        self.agent_num = len(self.UAV_og_positions)
        self.target_num = len(self.UAV_targets)
        # self.obs_dim = 5
        # self.action_dim = 10
        self.action_dim = 27

        # 更新观测维度计算，包括智能体位置、目标位置和障碍物信息
        self.obs_dim = 3 * self.agent_num + 3 * self.target_num + len(self.scene_blocks) * 6  # 每个障碍物有6个维度 --> 30

        # self.obs_dim = self.agent_num + self.target_num + len(self.scene_blocks) * 6  # 每个障碍物有6个维度

        self.action_mapping = {
            0: (-1, -1, -1),  # 后左上
            1: (-1, -1, 0),   # 左上
            2: (-1, -1, 1),   # 前左上
            3: (-1, 0, -1),   # 后上
            4: (-1, 0, 0),    # 上
            5: (-1, 0, 1),    # 前上
            6: (-1, 1, -1),   # 后右上
            7: (-1, 1, 0),    # 右上
            8: (-1, 1, 1),    # 前右上
            9: (0, -1, -1),   # 后左
            10: (0, -1, 0),   # 左
            11: (0, -1, 1),   # 前左
            12: (0, 0, -1),   # 后
            13: (0, 0, 0),    # 不移动
            14: (0, 0, 1),    # 前
            15: (0, 1, -1),   # 后右
            16: (0, 1, 0),    # 右
            17: (0, 1, 1),    # 前右
            18: (1, -1, -1),  # 后左下
            19: (1, -1, 0),   # 左下
            20: (1, -1, 1),   # 前左下
            21: (1, 0, -1),   # 后下
            22: (1, 0, 0),    # 下
            23: (1, 0, 1),    # 前下
            24: (1, 1, -1),   # 后右下
            25: (1, 1, 0),    # 右下
            26: (1, 1, 1)     # 前右下
        }




    def load_scene_data(self, filepath):
        with open(filepath, 'r') as file:
            self.scene_data = json.load(file)

    def load_targets(filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data['targets']
    
    
          
    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """
        # sub_agent_obs = []
        # for i in range(self.agent_num):
        #     sub_obs = np.random.random(size=(14,))
        #     sub_agent_obs.append(sub_obs)
        # return sub_agent_obs
    
        # # Lognam's testbed, ver1
        # # the problem is the size during env_runner is not compatiable
        # self.UAV_positions = np.array(self.UAV_og_positions)
        # sub_agent_obs = []
        # for i in range(self.agent_num-1):
        #     # 观测包括智能体位置、目标位置和障碍物信息
        #     obs = np.concatenate((self.UAV_positions.flatten(), self.UAV_targets.flatten(), self.scene_blocks.flatten()))
        #     sub_agent_obs.append(obs)
        # return sub_agent_obs
    
        # Lognam's testbed, ver2
        # 从字典中提取位置数据，并转换为NumPy数组
        self.UAV_positions = np.array([[pos['x'], pos['y'], pos['z']] for pos in self.UAV_og_positions])
        self.UAV_targets = np.array([[target['x'], target['y'], target['z']] for target in self.UAV_targets])
        
        # 处理障碍物信息
        blocks_data = []
        for block in self.scene_info['blocks']:
            block_pos_size = block['bottomCorner'] + block['size']
            blocks_data.extend(block_pos_size + [block['height']])
        
        # self.scene_blocks = np.array(blocks_data)

        sub_agent_obs = []
        for i in range(self.agent_num-1):
            # 观测包括智能体位置、目标位置和障碍物信息
            obs = np.concatenate((self.UAV_positions.flatten(), self.UAV_targets.flatten(), self.scene_blocks.flatten()))
            sub_agent_obs.append(obs)
        
        return np.array(sub_agent_obs)

    # def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
        # When self.agent_num is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
        # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
        """

        # print(actions)

        # sub_agent_obs = []
        # sub_agent_reward = []
        # sub_agent_done = []
        # sub_agent_info = []
        # for i in range(self.agent_num):
        #     sub_agent_obs.append(np.random.random(size=(14,)))
        #     sub_agent_reward.append([np.random.rand()])
        #     sub_agent_done.append(False)
        #     sub_agent_info.append({})

        # return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
    
        collision_distance = 1.5  # 定义碰撞距离阈值
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = [False] * self.agent_num
        sub_agent_info = [{}] * self.agent_num

        # 计算新的位置
        # new_positions = [self.UAV_positions[i] + np.array(self.action_mapping[actions[i]]) for i in range(self.agent_num)]
        new_positions = [self.UAV_positions[i] + np.array(self.action_mapping[int(actions[i])]) for i in range(self.agent_num)]


        # 检查碰撞
        for i in range(self.agent_num):
            for j in range(i + 1, self.agent_num):
                if np.linalg.norm(new_positions[i] - new_positions[j]) < collision_distance:
                    sub_agent_reward[i] = -10
                    sub_agent_reward[j] = -10
                    sub_agent_info[i]['collision'] = 'with UAV'
                    sub_agent_info[j]['collision'] = 'with UAV'

        # 检查与障碍物的碰撞和目标覆盖
        for i, pos in enumerate(new_positions):
            for block in self.scene_blocks:
                block_corner = np.array(block[:3])
                block_size = np.array(block[3:6])
                if np.all(pos >= block_corner) and np.all(pos < block_corner + block_size):
                    sub_agent_reward[i] -= 10
                    sub_agent_info[i]['collision'] = 'with block'

            # 检查目标覆盖
            for idx, target in enumerate(self.UAV_targets):
                if np.linalg.norm(pos - target) < 1.0:  # 假设覆盖半径为1
                    if idx not in self.covered_targets:
                        self.covered_targets.append(idx)
                        sub_agent_reward[i] += 10  # 覆盖新目标奖励
                    break

        # 更新智能体位置
        for i in range(self.agent_num):
            self.UAV_positions[i] = new_positions[i]

        # 生成观测数据
        for i in range(self.agent_num):
            obs = np.concatenate((self.UAV_positions[i], self.UAV_targets.flatten(), self.scene_blocks.flatten()))
            sub_agent_obs.append(obs)

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]


    # def step(self, actions):
        collision_distance = 1.5  # 定义碰撞距离阈值
        goal_tolerance = 1.0  # 定义接近目标的容忍距离

        # new_positions = [self.UAV_positions[i] + np.array(self.action_mapping[actions[i]]) for i in range(self.agent_num)]
        new_positions = [self.UAV_positions[i] + np.array(self.action_mapping[int(actions[i])]) for i in range(self.agent_num)]


        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = [False] * self.agent_num
        sub_agent_info = [{}] * self.agent_num

        # 检查碰撞（包括与其他UAV和障碍物）
        for i in range(self.agent_num):
            current_position = new_positions[i]

            # 检查与其他UAV的碰撞
            for j in range(self.agent_num):
                if i != j and np.linalg.norm(new_positions[i] - new_positions[j]) < collision_distance:
                    sub_agent_reward[i] = -10
                    sub_agent_info[i]['collision'] = 'with UAV'
                    sub_agent_done[i] = True

            # 检查与障碍物的碰撞
            for block in self.scene_blocks:
                block_corner = np.array(block[:3])
                block_size = np.array(block[3:6])
                if np.all(current_position >= block_corner) and np.all(current_position < block_corner + block_size):
                    sub_agent_reward[i] -= 10
                    sub_agent_info[i]['collision'] = 'with block'
                    sub_agent_done[i] = True

            # 检查是否接近任一目标
            for target in self.UAV_targets:
                if np.linalg.norm(current_position - target) <= goal_tolerance:
                    sub_agent_reward[i] += 10  # 接近目标时给予正奖励
                    break

        # 更新智能体位置
        self.UAV_positions = new_positions

        # 更新观测数据
        for i in range(self.agent_num):
            obs = np.concatenate((self.UAV_positions[i], self.UAV_targets.flatten(), self.scene_blocks.flatten()))
            sub_agent_obs.append(obs)

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    # def step(self, actions):
        collision_distance = 1.5  # 碰撞距离阈值
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = [False] * self.agent_num
        sub_agent_info = [{}] * self.agent_num

        # 暂存新位置，以便先计算所有新位置再更新
        new_positions = [None] * self.agent_num

        # 为每个智能体选择最优动作并计算新位置
        for i in range(self.agent_num):
            current_position = self.UAV_positions[i]
            best_action = None
            min_distance = float('inf')

            # 探索所有可能的动作以找到最优解
            for action, movement in self.action_mapping.items():
                potential_position = current_position + np.array(movement)
                distance = np.linalg.norm(potential_position - self.UAV_targets[i])

                if distance < min_distance:
                    # 检查潜在位置与其他智能体的距离
                    collision = False
                    for j in range(self.agent_num):
                        if i != j and np.linalg.norm(potential_position - self.UAV_positions[j]) < collision_distance:
                            collision = True
                            break

                    # # 检查潜在位置与障碍物的碰撞
                    for block in self.scene_blocks:
                        block_corner = np.array(block[:3])
                        block_size = np.array(block[3:6])
                        if np.all(potential_position >= block_corner) and np.all(potential_position < block_corner + block_size):
                            collision = True
                            break


                    # for block in self.scene_blocks:
                    #     block_corner = np.array(block[:3])
                    #     block_size = np.array(block[3:6])
                    #     if np.all(potential_position >= block_corner) and np.all(potential_position < block_corner + block_size):
                    #         collision = True
                    #         break

                    if not collision:
                        min_distance = distance
                        best_action = action
                        new_positions[i] = potential_position

            print("i is "+str(i))
            print("reward are "+str(sub_agent_reward))

            if best_action is not None:
                sub_agent_reward[i] = 10 - min_distance  # 奖励是距离目标的反比
            else:
                sub_agent_reward[i] = -10  # 如果没有合适的动作则给予负奖励
                sub_agent_done[i] = True
                sub_agent_info[i]['blocked'] = True

        # 现在更新智能体位置
        for i in range(self.agent_num):
            if new_positions[i] is not None:
                self.UAV_positions[i] = new_positions[i]

        # 更新观测数据
        for i in range(self.agent_num):
            obs = np.concatenate((self.UAV_positions[i], self.UAV_targets.flatten(), self.scene_blocks.flatten()))
            sub_agent_obs.append(obs)

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    def step(self, actions):
        collision_distance = 1.5  # 碰撞距离阈值
        sub_agent_obs = []
        sub_agent_reward = [0] * self.agent_num  # 初始化每个智能体的奖励为0
        sub_agent_done = [False] * self.agent_num
        sub_agent_info = [{}] * self.agent_num

        new_positions = [None] * self.agent_num  # 存储每个智能体的新位置

        for i in range(self.agent_num):
            current_position = self.UAV_positions[i]
            best_action = None
            min_distance = float('inf')

            # 探索所有可能的动作以找到最优解
            for action, movement in self.action_mapping.items():
                potential_position = current_position + np.array(movement)
                distance = np.linalg.norm(potential_position - self.UAV_targets[i])

                if distance < min_distance:
                    collision = False

                    # 检查与其他智能体的碰撞
                    for j in range(self.agent_num):
                        if i != j and np.linalg.norm(potential_position - self.UAV_positions[j]) < collision_distance:
                            collision = True
                            break

                    # 检查与障碍物的碰撞
                    for block in self.scene_blocks:
                        block_corner = block[:3]
                        block_size = block[3:6]
                        if np.all(potential_position >= block_corner) and np.all(potential_position < block_corner + block_size):
                            collision = True
                            break

                    if not collision:
                        min_distance = distance
                        best_action = action
                        new_positions[i] = potential_position

            # 根据找到的最佳动作更新奖励和状态
            if best_action is not None:
                sub_agent_reward[i] = 10 - min_distance  # 奖励是距离目标的反比
            else:
                sub_agent_reward[i] = -10  # 如果没有合适的动作则给予负奖励
                sub_agent_done[i] = True
                sub_agent_info[i]['blocked'] = True

        # 更新智能体位置
        for i in range(self.agent_num):
            if new_positions[i] is not None:
                self.UAV_positions[i] = new_positions[i]

        # 更新观测数据
        for i in range(self.agent_num):
            obs = np.concatenate((self.UAV_positions[i], self.UAV_targets.flatten(), self.scene_blocks.flatten()))
            sub_agent_obs.append(obs)

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
