from cmath import pi
import math
import numpy as np
from gym_usv.envs.usv_asmc_ye_int_env import UsvAsmcYeIntEnv



env = UsvAsmcYeIntEnv()
env.reset()
for _ in range(10000):
    env.render()
    env.step(pi/4) # take a random action
env.close()
