import argparse
from pathlib import Path
from functools import partial
import inspect
import torch
import gymnasium as gym
import pystk2_gymnasium  # IMPORTANT
from stable_baselines3 import SAC, PPO
import numpy as np

from pystk2_gymnasium import AgentSpec
from bbrl.agents.gymnasium import ParallelGymAgent, make_env

from .pystk_actor import env_name, get_wrappers, player_name


def train(algorithm: str, init_method: str, total_timesteps: int, tb_log_dir: str):
    make_stkenv = partial(
        make_env,
        env_name,
        wrappers=get_wrappers(),
        render_mode=None,
        autoreset=True,
        num_kart=2,
        agent=AgentSpec(use_ai=False, name=player_name),
    )
    env_agent = ParallelGymAgent(make_stkenv, 1)
    env = env_agent.envs[0]
    print("Obs space:", env.observation_space)
    print("Act space:", env.action_space)

    policy_kwargs = dict(
        net_arch=[256, 256],  
        activation_fn=torch.nn.ReLU,
    )

    # Sélection de l'algorithme
    if algorithm.upper() == "SAC":
        model_class = SAC
        tb_log_name = "sac_reward"
        tensorboard_log = f"{tb_log_dir}/sac_{init_method}"
    elif algorithm.upper() == "PPO":
        model_class = PPO
        tb_log_name = "ppo_reward"
        tensorboard_log = f"{tb_log_dir}/ppo_{init_method}"
    else:
        raise ValueError(f"Algorithme non supporté : {algorithm}")

    # Initialisation du modèle
    model = model_class(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tensorboard_log
    )

    if init_method == "bc":
        bc_path = Path("./stk_actor/bc_model.pth")
        if not bc_path.exists():
            raise FileNotFoundError(f"Le modèle BC pré-entraîné est introuvable à l'emplacement : {bc_path}. Veuillez d'abord exécuter train_BC_model.py.")
        
        print(f"Chargement des poids depuis {bc_path}...")
        bc_state = torch.load(bc_path, map_location="cpu")
        target_actor = model.policy.actor if algorithm.upper() == "SAC" else model.policy

        with torch.no_grad():
            for target_param, bc_param in zip(
                target_actor.parameters(),
                bc_state.values(),
            ):
                target_param.copy_(bc_param)

        print("Politique initialisée avec succès depuis le modèle BC.")
    else:
        print("Politique initialisée de manière aléatoire.")

    print(f"Début de l'apprentissage ({algorithm.upper()}) pour {total_timesteps} pas de temps...")
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=f"{tb_log_name}_{init_method}_init",
    )

    # Sauvegarde du modèle entraîné
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    if algorithm.upper() == "SAC" and init_method == "bc":
        output_filename = "pystk_actor.pth"
    else:
        output_filename = f"pystk_actor_{algorithm.lower()}_{init_method}.pth"
        
    save_path = mod_path / output_filename
    torch.save(model.policy.state_dict(), save_path)
    print(f"Politique optimale sauvegardée sous : {save_path}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script unifié d'entraînement pour SuperTuxKart RL")
    parser.add_argument(
        "--algo",
        type=str,
        choices=["sac", "ppo"],
        default="sac",
        help="Algorithme d'apprentissage par renforcement à utiliser (sac ou ppo)"
    )
    parser.add_argument(
        "--init",
        type=str,
        choices=["bc", "alea"],
        default="bc",
        help="Méthode d'initialisation des poids (bc pour Behavioral Cloning, alea pour aléatoire)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Nombre total de pas de temps pour l'entraînement"
    )
    parser.add_argument(
        "--tb-log-dir",
        type=str,
        default="./tb_logs",
        help="Dossier de sauvegarde des logs TensorBoard"
    )

    args = parser.parse_args()
    train(
        algorithm=args.algo,
        init_method=args.init,
        total_timesteps=args.timesteps,
        tb_log_dir=args.tb_log_dir
    )
