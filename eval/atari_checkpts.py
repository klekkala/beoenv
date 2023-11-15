
#g1_TCN = '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_TCN_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_4.0_0.01_32_32_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_12_11_47_47/checkpoint/'
#g2_TCN = '/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_1CHAN_TCN_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_4.0_0.01_32_32_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_12_11_48_05/checkpoint/'




DA_ckpts = {'DA_e2e': '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_e2e_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_11_15_10_22/checkpoint/',
'DA_SD_SOM': '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_SOM_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_0.9_32_32_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_11_03_06_22_07/checkpoint/',
'DA_SD_TCN': '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_TCN_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_1.0_0.01_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_16_12_52_22/checkpoint/',
'DA_SD_VIP': '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_VIP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_50.0_15.0_0.01_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_16_22_29_43/checkpoint/',
'DA_SD_VEP': '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_VEP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_1.0_0.1_1.0_same_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_19_18_54_18/checkpoint/',
'DA_SD_NVEP': '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_NVEP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_1.0_0.1_1.0_same_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_11_13_19_05_11/checkpoint/',
'DA_SCD_VEP': '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_VEP_ATARI_EXPERT_1CHAN_SPACECARNIVALDEMO_STANDARD_1.0_0.1_1.0_same_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_26_12_05_45/checkpoint/',
'DA_random': '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_random_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_17_18_27_30/checkpoint/'}


SA_ckpts = {'SA_e2e': '/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_e2e_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_11_15_12_08/checkpoint/',
'SA_SD_SOM': '/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_1CHAN_SOM_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_0.9_32_32_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_11_03_06_20_13/checkpoint/',
'SA_SD_TCN': '/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_1CHAN_TCN_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_1.0_0.01_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_27_01_20_55/checkpoint/',
'SA_SD_VIP': '/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_1CHAN_VIP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_50.0_15.0_0.01_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_16_22_29_56/checkpoint/',
'SA_SD_VEP': '/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_1CHAN_VEP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_1.0_0.1_1.0_same_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_19_18_53_54/checkpoint/',
'SA_SD_NVEP': '/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_1CHAN_NVEP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_1.0_0.1_1.0_same_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_11_13_03_45_07/checkpoint/',
'SA_SCD_VEP': '/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_1CHAN_VEP_ATARI_EXPERT_1CHAN_SPACECARNIVALDEMO_STANDARD_1.0_0.1_1.0_same_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_26_12_05_07/checkpoint/',
'SA_random': '/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_random_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_17_13_36_39/checkpoint/'}


models = [DA_ckpts, SA_ckpts]