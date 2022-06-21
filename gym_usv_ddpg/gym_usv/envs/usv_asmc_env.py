"""
@author: Alejandro Gonzalez 

Environment of an Unmanned Surface Vehicle with an
Adaptive Sliding Mode Controller to train guidance
laws on the OpenAI Gym library.
"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class UsvAsmcEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        #Integral step (or derivative) for 100 Hz
        # 因为ddpg更新一次（step=0.05s），asmc更新5次(0.01s)
        self.integral_step = 0.01
        self.min_speed = 0.3

        #USV model coefficients
        self.X_u_dot = -2.25
        self.Y_v_dot = -23.13
        self.Y_r_dot = -1.31
        self.N_v_dot = -16.41
        self.N_r_dot = -2.79
        self.Xuu = 0
        self.Yvv = -99.99
        self.Yvr = -5.49
        self.Yrv = -5.49
        self.Yrr = -8.8
        self.Nvv = -5.49
        self.Nvr = -8.8
        self.Nrv = -8.8
        self.Nrr = -3.49
        self.m = 30
        self.Iz = 4.1
        self.B = 0.41
        self.c = 0.78

        #ASMC gain
        self.k_u = 0.1
        self.k_psi = 0.2
        self.kmin_u = 0.05
        self.kmin_psi = 0.2
        self.k2_u = 0.02
        self.k2_psi = 0.1
        self.mu_u = 0.05
        self.mu_psi = 0.1
        self.lambda_u = 0.001
        self.lambda_psi = 1

        self.k_ak = 5.72
        self.k_ye = 0.5
        self.sigma_ye = 1.

        self.state = None
        self.velocity = None
        self.position = None
        self.aux_vars = None
        self.last = None
        self.target = None

        self.max_y = 10
        self.min_y = -10
        self.max_x = 30
        self.min_x = -10

        self.viewer = None

        self.min_action = -np.pi/2
        self.max_action = np.pi/2

        self.c_action = 1. / np.power((self.max_action/2-self.min_action/2)/self.integral_step, 2)
        self.w_action = 0.2

        self.min_uv = -1.5
        self.max_uv = 1.5
        self.min_r = -1.
        self.max_r = 1.
        self.min_ye = -10
        self.max_ye = 10
        self.min_psi_ak = -np.pi
        self.max_psi_ak = np.pi

        self.low_state = np.array([self.min_uv, self.min_uv, self.min_r, self.min_ye, self.min_psi_ak, self.min_action], dtype=np.float32)
        self.high_state = np.array([self.max_uv, self.max_uv, self.max_r, self.max_ye, self.max_psi_ak, self.max_action], dtype=np.float32)
        # ye的导数没有？
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(1,), dtype=np.float32)
        # 在Gym中，离散空间一般用gym.spaces.Discrete类表示，连续空间一般用gym.spaces.Box类表示，shape=(1,)
        # 表示创建一维连续动作空间 每个维度相同边界 shape=None即空对象，占位
        # Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        # 二维 ，每个维度独立边界 [-1.0,2.0] [-2.0,4.0]
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)


    def step(self, action):
        state = self.state
        velocity = self.velocity
        position = self.position
        aux_vars = self.aux_vars
        last = self.last
        target = self.target


        u, v_ak, r_ak, ye, psi_ak, action_last = state
        u, v, r = velocity
        x, y, psi = position
        e_u_int, Ka_u, Ka_psi = aux_vars  # 浪涌速度误差，K_i(u),K_i(psi)即自适应增益
        x_dot_last, y_dot_last, psi_dot_last, u_dot_last, v_dot_last, r_dot_last, e_u_last, Ka_dot_u_last, Ka_dot_psi_last = last
        x_0, y_0, desired_speed, ak, x_d, y_d = target

        eta = np.array([x, y, psi])
        upsilon = np.array([u, v, r])
        eta_dot_last = np.array([x_dot_last, y_dot_last, psi_dot_last])
        upsilon_dot_last = np.array([u_dot_last, v_dot_last, r_dot_last])

        action_dot = (action - action_last)/self.integral_step
        action_last = action

        psi_d = action + ak # psi_d是期望航向角，ak是pp角， action是  ak = 0
        # ？
        psi_d = np.where(np.greater(np.abs(psi_d), np.pi), (np.sign(psi_d))*(np.abs(psi_d)-2*np.pi), psi_d)

        Xu = -25
        Xuu = 0
        if(abs(upsilon[0]) > 1.2):
            Xu = 64.55
            Xuu = -70.92 # table2 ,X_u_abs(u),u是浪涌速度

        Yv = 0.5*(-40*1000*abs(upsilon[1])) * \
            (1.1+0.0045*(1.01/0.09) - 0.1*(0.27/0.09)+0.016*(np.power((0.27/0.09), 2)))
        Yr = 6*(-3.141592*1000) * \
            np.sqrt(np.power(upsilon[0], 2)+np.power(upsilon[1], 2))*0.09*0.09*1.01
        Nv = 0.06*(-3.141592*1000) * \
            np.sqrt(np.power(upsilon[0], 2)+np.power(upsilon[1], 2))*0.09*0.09*1.01
        Nr = 0.02*(-3.141592*1000) * \
            np.sqrt(np.power(upsilon[0], 2)+np.power(upsilon[1], 2))*0.09*0.09*1.01*1.01

        g_u = 1 / (self.m - self.X_u_dot) # 公式10
        g_psi = 1 / (self.Iz - self.N_r_dot) # 公式20

        f_u = (((self.m - self.Y_v_dot)*upsilon[1]*upsilon[2] + (Xuu*np.abs(upsilon[0]) + Xu*upsilon[0])) / (self.m - self.X_u_dot)) # 公式9
        f_psi = (((-self.X_u_dot + self.Y_v_dot)*upsilon[0]*upsilon[1] + (Nr * upsilon[2])) / (self.Iz - self.N_r_dot)) #公式19

        e_psi = psi_d - eta[2] #期望航向和实际航向误差
        e_psi = np.where(np.greater(np.abs(e_psi), np.pi), (np.sign(e_psi))*(np.abs(e_psi)-2*np.pi), e_psi) # ？
        e_psi_dot = 0 - upsilon[2] # 误差角速度

        abs_e_psi = np.abs(e_psi)

        u_psi = 1/(1 + np.exp(10*(abs_e_psi*(2/np.pi) - 0.5))) # ？

        u_d_high = (desired_speed - self.min_speed)*u_psi + self.min_speed # 最大速度限制范围
        u_d = u_d_high # 变换后的期望速度

        e_u = u_d - upsilon[0] # 公式12,速度误差
        e_u_int = self.integral_step*(e_u + e_u_last)/2 + e_u_int # e_u的积分

        sigma_u = e_u + self.lambda_u * e_u_int  # 滑模向量s_u ,公式13
        sigma_psi = e_psi_dot + self.lambda_psi * e_psi # 公式23 ，dot是导数的意思 dot(a,b)是矩阵乘法

        Ka_dot_u = np.where(np.greater(Ka_u, self.kmin_u), self.k_u * np.sign(np.abs(sigma_u) - self.mu_u), self.kmin_u) # 公式17,K_i的导数
        Ka_dot_psi = np.where(np.greater(Ka_psi, self.kmin_psi), self.k_psi * np.sign(np.abs(sigma_psi) - self.mu_psi), self.kmin_psi) #对应17的psi

        Ka_u = self.integral_step*(Ka_dot_u + Ka_dot_u_last)/2 + Ka_u # 浪涌速度的自适应增益K_i
        Ka_dot_u_last = Ka_dot_u

        Ka_psi = self.integral_step*(Ka_dot_psi + Ka_dot_psi_last)/2 + Ka_psi # 航向角psi的自适应增益K_i
        Ka_dot_psi_last = Ka_dot_psi

        ua_u = (-Ka_u * np.power(np.abs(sigma_u), 0.5) * np.sign(sigma_u)) - (self.k2_u * sigma_u) #公式16,sigma_u是s_u(滑模量s)
        ua_psi = (-Ka_psi * np.power(np.abs(sigma_psi), 0.5) * np.sign(sigma_psi)) - (self.k2_psi * sigma_psi) # 对应16的psi（航向角）

        Tx = ((self.lambda_u * e_u) - f_u - ua_u) / g_u  
        # e_u是u_d - u 即浪涌速度误差,公式15
        #ua_u is taken by the adaptive sliding mode strategy, 期望速度的导数呢?
        Tz = ((self.lambda_psi * e_psi) - f_psi - ua_psi) / g_psi

        Tport = (Tx / 2) + (Tz / self.B)
        Tstbd = (Tx / (2*self.c)) - (Tz / (self.B*self.c))

        Tport = np.where(np.greater(Tport, 36.5), 36.5, Tport)
        # np.where(条件，x,y) 满足条件输出x   np.greater(x,y) x>y输出true
        Tport = np.where(np.less(Tport, -30), -30, Tport)
        Tstbd = np.where(np.greater(Tstbd, 36.5), 36.5, Tstbd)
        Tstbd = np.where(np.less(Tstbd, -30), -30, Tstbd)
        # 船模型的质量矩阵
        M = np.array([[self.m - self.X_u_dot, 0, 0],
                      [0, self.m - self.Y_v_dot, 0 - self.Y_r_dot],
                      [0, 0 - self.N_v_dot, self.Iz - self.N_r_dot]])
        # 推力向量[Tx,Ty,Tpsi]
        T = np.array([Tport + self.c*Tstbd, 0, 0.5*self.B*(Tport - self.c*Tstbd)])

        CRB = np.array([[0, 0, 0 - self.m * upsilon[1]],
                        [0, 0, self.m * upsilon[0]],
                        [self.m * upsilon[1], 0 - self.m * upsilon[0], 0]])
        
        CA = np.array([[0, 0, 2 * ((self.Y_v_dot*upsilon[1]) + ((self.Y_r_dot + self.N_v_dot)/2) * upsilon[2])],
                       [0, 0, 0 - self.X_u_dot * self.m * upsilon[0]],
                       [2*(((0 - self.Y_v_dot) * upsilon[1]) - ((self.Y_r_dot+self.N_v_dot)/2) * upsilon[2]), self.X_u_dot * self.m * upsilon[0], 0]])
        # Coriolis 矩阵C(v)
        C = CRB + CA

        Dl = np.array([[0 - Xu, 0, 0],
                       [0, 0 - Yv, 0 - Yr],
                       [0, 0 - Nv, 0 - Nr]])

        Dn = np.array([[Xuu * abs(upsilon[0]), 0, 0],
                       [0, self.Yvv * abs(upsilon[1]) + self.Yvr * abs(upsilon[2]), self.Yrv *
                        abs(upsilon[1]) + self.Yrr * abs(upsilon[2])],
                       [0, self.Nvv * abs(upsilon[1]) + self.Nvr * abs(upsilon[2]), self.Nrv * abs(upsilon[1]) + self.Nrr * abs(upsilon[2])]])
        # D(v)矩阵
        D = Dl - Dn
        # upsilon=[u,v,r] 公式1求其导数然后积分求出upsilon inv是求逆矩阵
        upsilon_dot = np.matmul(np.linalg.inv(
            M), (T - np.matmul(C, upsilon) - np.matmul(D, upsilon)))
        upsilon = (self.integral_step) * (upsilon_dot +
                                               upsilon_dot_last)/2 + upsilon  # integral
        upsilon_dot_last = upsilon_dot
        # 旋转矩阵R(yta)
        J = np.array([[np.cos(eta[2]), -np.sin(eta[2]), 0],
                      [np.sin(eta[2]), np.cos(eta[2]), 0],
                      [0, 0, 1]])
        # 公式2
        eta_dot = np.matmul(J, upsilon)  # transformation into local reference frame
        eta = (self.integral_step)*(eta_dot+eta_dot_last)/2 + eta  # integral
        eta_dot_last = eta_dot

        psi = eta[2]
        psi = np.where(np.greater(np.abs(psi), np.pi), (np.sign(psi))*(np.abs(psi)-2*np.pi), psi)
        # 相对于pp角的转向误差
        psi_ak = psi - ak
        psi_ak = np.where(np.greater(np.abs(psi_ak), np.pi), (np.sign(psi_ak))*(np.abs(psi_ak)-2*np.pi), psi_ak)
        # 公式28 ye > 0在直线的右边，ye < 0在左边
        ye = -(eta[0] - x_0)*np.math.sin(ak) + (eta[1] - y_0)*np.math.cos(ak)
        ye_abs = np.abs(ye)

        reward = self.compute_reward(ye_abs, psi_ak, action_dot)
        # 由u,v转换到x_dot和y_dot  pp角a_k是旋转（转换）矩阵的输入yita=psi_ak
        u_ak, v_ak = self.body_to_path(upsilon[0], upsilon[1], psi_ak)
       
        if ye_abs > self.max_ye or eta[0] < self.min_x:
            done = True
            reward = -1
        else:
            done = False
        # v_a_k是y_e_dot？ 不需要矩阵转换吗？
        self.state = np.array([upsilon[0], v_ak, upsilon[2], ye, psi_ak, action_last])
        self.velocity = np.array([upsilon[0], upsilon[1], upsilon[2]])
        self.position = np.array([eta[0], eta[1], psi])
        self.aux_vars = np.array([e_u_int, Ka_u, Ka_psi])
        self.last = np.array([eta_dot_last[0], eta_dot_last[1], eta_dot_last[2], upsilon_dot_last[0], upsilon_dot_last[1], upsilon_dot_last[2], e_u_last, Ka_dot_u_last, Ka_dot_psi_last])
        # 将六维状态变为向量
        state = self.state.reshape(self.observation_space.shape[0])

        return state, reward, done, {}

    
    def reset(self):
        # 初始位置均匀采样，初始速度为0
        x = np.random.uniform(low=-2.5, high=2.5)
        y = np.random.uniform(low=-2.5, high=2.5)
        psi = np.random.uniform(low=-np.pi, high=np.pi)
        eta = np.array([x, y])
        upsilon = np.array([0.,0.,0.])
        eta_dot_last = np.array([0.,0.,0.])
        upsilon_dot_last = np.array([0.,0.,0.])
        e_u_int = 0.
        Ka_u = 0.
        Ka_psi = 0.
        e_u_last = 0.
        Ka_dot_u_last = 0.
        Ka_dot_psi_last = 0.
        action_last = 0.
        # p_k  直线路径
        x_0 = np.random.uniform(low=-2.5, high=2.5)
        y_0 = np.random.uniform(low=-2.5, high=2.5)
        # p_k+1  直线路径
        x_d = np.random.uniform(low=15, high=30)
        y_d = y_0
        # 期望速度均匀采样
        desired_speed = np.random.uniform(low=0.4, high=1.4)
        # numpy.arctan与math.atan的结果的取值范围是一样的，是从-90度到90度，而math.atan2的结果的取值范围是从-180到180度（反正切arctan）
        # 因为画的竖直线，ak = 0
        ak = np.math.atan2(y_d-y_0,x_d-x_0)
        ak = np.float32(ak)
       

        psi_ak = psi - ak
        psi_ak = np.where(np.greater(np.abs(psi_ak), np.pi), np.sign(psi_ak)*(np.abs(psi_ak)-2*np.pi), psi_ak)
        psi_ak = np.float32(psi_ak)
        ye = -(x - x_0)*np.math.sin(ak) + (y - y_0)*np.math.cos(ak)

        u_ak, v_ak = self.body_to_path(upsilon[0], upsilon[1], psi_ak)

        self.state = np.array([upsilon[0], v_ak, upsilon[2], ye, psi_ak, action_last])
        self.velocity = np.array([upsilon[0], upsilon[1], upsilon[2]])
        self.position = np.array([eta[0], eta[1], psi])
        self.aux_vars = np.array([e_u_int, Ka_u, Ka_psi])
        self.last = np.array([eta_dot_last[0], eta_dot_last[1], eta_dot_last[2], upsilon_dot_last[0], upsilon_dot_last[1], upsilon_dot_last[2], e_u_last, Ka_dot_u_last, Ka_dot_psi_last])
        self.target = np.array([x_0, y_0, desired_speed, ak, x_d, y_d])
        
        state = self.state.reshape(self.observation_space.shape[0])

        return state

    # 渲染
    def render(self, mode='human'):

        screen_width = 400
        screen_height = 800

        world_width = self.max_y - self.min_y
        scale = screen_width/world_width
        boat_width = 15
        boat_height = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            clearance = 10
            l, r, t, b = -boat_width/2, boat_width/2, boat_height, 0
            boat = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            boat.add_attr(rendering.Transform(translation=(0, clearance)))
            self.boat_trans = rendering.Transform()
            boat.add_attr(self.boat_trans)
            self.viewer.add_geom(boat)

        x_0 = (self.target[0] - self.min_x)*scale
        y_0 = (self.target[1] - self.min_y)*scale
        x_d = (self.target[4] - self.min_x)*scale
        y_d = (self.target[5] - self.min_y)*scale
        start = (y_0, x_0)
        end = (y_d, x_d)

        self.viewer.draw_line(start, end)

        x = self.position[0]
        y = self.position[1]
        psi = self.position[2]

        self.boat_trans.set_translation((y-self.min_y)*scale, (x-self.min_x)*scale)
        self.boat_trans.set_rotation(-psi)

        self.viewer.draw_line(start, end)

        return self.viewer.render(return_rgb_array = mode == 'rgb_array')

    def close(self):

        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def compute_reward(self, ye, psi_ak, action_dot):

        psi_ak = np.abs(psi_ak)

        reward_action = self.w_action*np.math.tanh(-self.c_action*np.power(action_dot, 2))
        # sigma_ye = 1
        reward_ye = np.where(np.greater(ye, self.sigma_ye), np.exp(-self.k_ye*ye), np.exp(-self.k_ye*np.power(ye, 2)/self.sigma_ye))
        reward_ak = -np.exp(self.k_ak*(psi_ak - np.pi))

        reward = np.where(np.less(psi_ak, np.pi/2), reward_action + reward_ye, reward_ak)
        return reward
    # u,v -> x_dot,y_dot 
    def body_to_path(self, x2, y2, alpha):
        '''
        @name: body_to_path
        @brief: Coordinate transformation between body and path reference frames.
        @param: x2: target x coordinate in body reference frame
                y2: target y coordinate in body reference frame
        @return: path_x2: target x coordinate in path reference frame
                 path_y2: target y coordinate in path reference frame
        '''
        p = np.array([x2, y2])
        J = self.rotation_matrix(alpha)
        n = J.dot(p)
        path_x2 = n[0]
        path_y2 = n[1]
        return (path_x2, path_y2)
    # 未增维的旋转矩阵
    def rotation_matrix(self, angle):
        '''
        @name: rotation_matrix
        @brief: Transformation matrix template.
        @param: angle: angle of rotation
        @return: J: transformation matrix
        '''
        J = np.array([[np.math.cos(angle), -1*np.math.sin(angle)],
                      [np.math.sin(angle), np.math.cos(angle)]])
        return (J)