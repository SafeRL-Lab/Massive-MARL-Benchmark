# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from ast import arg
import numpy as np
import random

from agents.utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from agents.utils.parse_task import parse_task
from agents.utils.process_sarl import process_sarl
from agents.utils.process_marl import process_MultiAgentRL, get_AgentIndex
from agents.utils.process_mtrl import *
from agents.utils.process_metarl import *
from agents.utils.process_offrl import *

from config import get_config

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            
            from envs.env_continuous import ContinuousActionEnv

            env = ContinuousActionEnv()

            # from envs.env_discrete import DiscreteActionEnv

            # env = DiscreteActionEnv()

            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            from envs.env_continuous import ContinuousActionEnv

            env = ContinuousActionEnv()
            # from envs.env_discrete import DiscreteActionEnv
            # env = DiscreteActionEnv()
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="MyEnv", help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args
    


def train():
    print("Algorithm: ", args.algo)
    agent_index = get_AgentIndex(cfg)

    if args.algo in ["mappo", "happo", "hatrpo","maddpg","ippo"]: 
        # maddpg exists a bug now 
        args.task_type = "MultiAgent"

        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

        runner = process_MultiAgentRL(args,env=env, config=cfg_train, model_dir=args.model_dir)
        
        # test
        if args.model_dir != "":
            eval_info = runner.eval(1000)
            print("Evaluation Information:", eval_info)
        else:
            runner.run()

     if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))    


           

    elif args.algo in ["ppo","ddpg","sac","td3","trpo"]:
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

        sarl = eval('process_sarl')(args, env, cfg_train, logdir)

        iterations = cfg_train["learn"]["max_iterations"]
        if args.max_iterations > 0:
            iterations = args.max_iterations

        sarl.run(num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"])
    
    elif args.algo in ["mtppo", "random"]:
        args.task_type = "MultiTask"

        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

        mtrl = eval('process_{}'.format(args.algo))(args, env, cfg_train, logdir)

        iterations = cfg_train["learn"]["max_iterations"]
        if args.max_iterations > 0:
            iterations = args.max_iterations

        mtrl.run(num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"])

    elif args.algo in ["mamlppo"]:
        args.task_type = "Meta"
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

        trainer = eval('process_{}'.format(args.algo))(args, env, cfg_train, logdir)

        iterations = cfg_train["learn"]["max_iterations"]
        if args.max_iterations > 0:
            iterations = args.max_iterations

        trainer.train(train_epoch=iterations)

    elif args.algo in ["td3_bc", "bcq", "iql", "ppo_collect"]:
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

        offrl = eval('process_{}'.format(args.algo))(args, env, cfg_train, logdir)

        iterations = cfg_train["learn"]["max_iterations"]
        if args.max_iterations > 0:
            iterations = args.max_iterations

        offrl.run(num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"])

    else:
        print("Unrecognized algorithm!\nAlgorithm should be one of: [happo, hatrpo, mappo,ippo,maddpg,sac,td3,trpo,ppo,ddpg, mtppo, random, mamlppo, td3_bc, bcq, iql, ppo_collect]")


if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    train()
