from os.path import join as ospj

from NFSP.TrainingProfile import TrainingProfile
from NFSP.workers.driver.Driver import Driver
from PokerRL import LimitHoldem
from PokerRL.game.wrappers import FlatNonHULimitPokerEnvBuilder
from PokerRL.util import file_util
from NFSP.LimitObsInterpreter import LimitObsInterpreter
from PokerRL.eval.head_to_head.HistoryArgs import HistoryArgs

if __name__ == '__main__':
    t_prof=TrainingProfile(name="nine_player_limit",
                             nn_type="feedforward",
                             nn_structure="paper",
                             n_envs=256,
                             n_steps_pretrain_per_la=0,
                             n_steps_per_iter_per_la=256,
                             device_inference="cpu",
    
                             game_cls=LimitHoldem,
                             n_seats=9,
                             use_simplified_headsup_obs=False,
                             start_chips=48,
                             stack_randomization_range=(0, 0),
                             
                             feedforward_env_builder=FlatNonHULimitPokerEnvBuilder,
                             export_hands_freq=999999999,
                             eval_agent_export_freq=100,
                             DISTRIBUTED=False,
                             target_net_update_freq=300,
                             use_pre_layers_br=False,
                             use_pre_layers_avg=False,
                             TESTING=True,
                             n_units_final_br=128,
                             n_units_final_avg=128,
                             n_merge_and_table_layer_units_br=128,
                             n_merge_and_table_layer_units_avg=128,
                             first_and_third_units=1024,
                             second_and_fourth_units=512,
    
                             mini_batch_size_br_per_la=128,
                             mini_batch_size_avg_per_la=128,
                             
                             history_args=HistoryArgs(100, 100)
                             )
    ctrl = Driver(t_prof,
                  eval_methods={"history": 10},
                  n_iterations=250)
    ctrl.run()
    #obs_interp = LimitObsInterpreter(t_prof)
    #buffer = obs_interp.GetActionAndHandBuffer()
    #for i in range(1, 5):
    #    file_name = '/home/robert/poker_ai_data/testing/export_hands/nine_player_limit/{}/LearnerActor_0_General.pkl'.format(i)
    #    print(file_name)
    #    buffer.load_from_file(file_name)
    #    print(buffer._pub_obs_buffer)
    #    print(buffer._range_idxs_buffer)
    #    for j in range(buffer._top):
    #        obs_interp.InterpretObs(buffer._pub_obs_buffer[j].numpy())
    eval_dir = ospj(t_prof.path_agent_export_storage, str(t_prof.name))
    dirs = file_util.get_all_dirs_in_dir(eval_dir)
    print(dirs)
    