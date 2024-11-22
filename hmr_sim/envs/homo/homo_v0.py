
from hmr_sim.utils.vis import render_homo
from hmr_sim.envs.homo.base import BaseEnv

class HomoV0(BaseEnv):

    def __init__(self, config):
        super().__init__(config)


    def render(self, mode='human'):
        render_homo(self)
