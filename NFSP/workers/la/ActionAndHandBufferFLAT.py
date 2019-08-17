import pickle
import torch

from PokerRL.rl.buffers._circular_base import CircularBufferBase

class ActionAndHandBufferFLAT(CircularBufferBase):
    
    def __init__(self, env_bldr, max_size):
        super().__init__(env_bldr=env_bldr, max_size=max_size)
        
        self.storage_device = torch.device("cpu")

        self._pub_obs_buffer = None
        self._range_idxs_buffer = None
        
        self.reset()
        
    def add_step(self, pub_obs, range_idxs):
        self._pub_obs_buffer[self._top] = torch.from_numpy(pub_obs).to(
            device=self.storage_device)
        self._range_idxs_buffer[self._top] = torch.LongTensor(range_idxs)

        if self._size < self._max_size:
            self._size += 1
        else:
            print("Hand Buffer Overloaded")

        self._top = (self._top + 1) % self._max_size
        
    def state_dict(self):
        return {
            "pub_obs_buffer": self._pub_obs_buffer.cpu().clone(),
            "range_idxs_buffer": self._range_idxs_buffer.cpu().clone(),
            "size": self._size,
            "top": self._top
        }

    def load_from_file(self, file_path):
        with open(file_path, "rb") as pkl_file:
            state = pickle.load(pkl_file)['action_and_hand_buffer']
            print(state)
            self._pub_obs_buffer = state["pub_obs_buffer"]
            self._range_idxs_buffer = state["range_idxs_buffer"]
            self._size = state["size"]
            self._top = state["top"]
        
    def reset(self):
        super().reset()
        self._pub_obs_buffer = torch.empty(size=(self._max_size,
                                                 self._env_bldr.pub_obs_size),
                                           dtype=torch.float32,
                                           device=self.storage_device)
        self._range_idxs_buffer = torch.empty(size=(self._max_size, self._env_bldr.N_SEATS),
                                              dtype=torch.long, device=self.storage_device)