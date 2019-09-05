from os.path import join as ospj

from NFSP.TrainingProfile import TrainingProfile
from NFSP.workers.driver.Driver import Driver
from PokerRL import LimitHoldem
from PokerRL.game.wrappers import FlatNonHULimitPokerEnvBuilder
from PokerRL.util import file_util
from NFSP.LimitObsInterpreter import LimitObsInterpreter
from PokerRL.eval.head_to_head.HistoryArgs import HistoryArgs

if __name__ == '__main__':
    
    N_WORKERS = 45
    t_prof=TrainingProfile(name="nine_player_limit_canonical",
                           
                           DISTRIBUTED=True,
                           n_learner_actor_workers=N_WORKERS,
                           
                          nn_type="feedforward",
                          nn_structure="paper",
                          cir_buf_size_each_la=6e5 / N_WORKERS,
                          res_buf_size_each_la=6e6 / N_WORKERS,
                          min_prob_add_res_buf=0.25,
                          
                          n_envs=256,
                          n_steps_pretrain_per_la=0,
                          n_steps_per_iter_per_la=256,
                          device_inference="cpu",

                          game_cls=LimitHoldem,
                          n_seats=9,
                          use_simplified_headsup_obs=False,
                          start_chips=48,
                          stack_randomization_range=(0, 0),
                          canonical=True,
                         
                          feedforward_env_builder=FlatNonHULimitPokerEnvBuilder,
                          
                          checkpoint_freq=500,
                          export_hands_freq=999999999,
                          eval_agent_export_freq=1000,
                          lite_checkpoint=True,
                          lite_checkpoint_steps=128000,
                          
                          target_net_update_freq=1000,
                          first_and_third_units=1024,
                          second_and_fourth_units=512,
                          
                          eps_start=0.08,
                          eps_const=0.007,
                          eps_exponent=0.5,
                          eps_min=0.0,

                          mini_batch_size_br_per_la=256,
                          mini_batch_size_avg_per_la=256,
                          n_br_updates_per_iter=2,
                          n_avg_updates_per_iter=2,
                          
                          history_args=HistoryArgs(1000, 10000)
                          )
    ctrl = Driver(t_prof,
                  eval_methods={},#"history": 1000},
                  n_iterations=None,
                  iteration_to_import=27000,
                  name_to_import="nine_player_limit_canonical_")
    ctrl.run()
