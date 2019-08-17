from PokerRL.rl import rl_util
from NFSP.workers.la.ActionAndHandBufferFLAT import ActionAndHandBufferFLAT

class LimitObsInterpreter:
    def __init__(self, t_prof):
        self._t_prof = t_prof
        self._env_bldr = rl_util.get_env_builder(t_prof)
        self._env_wrapper = self._env_bldr.get_new_wrapper(is_evaluating=False)
    
    def GetActionAndHandBuffer(self):
        return ActionAndHandBufferFLAT(self._env_bldr, 
                                       self._t_prof.action_and_hand_buffer_size)
    
    def InterpretObs(self, obs):
        self._env_wrapper.print_obs(obs)