from pathlib import Path
from functools import partial
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import pystk2_gymnasium  # IMPORTANT
from stable_baselines3 import SAC
from gymnasium.wrappers import FlattenObservation
import numpy as np

from pystk2_gymnasium import AgentSpec
from bbrl.agents.gymnasium import ParallelGymAgent, make_env

from .pystk_actor import env_name, get_wrappers, player_name
from .actors import ContinuousObs
from pystk2_gymnasium import ConstantSizedObservations

from torch.utils.data import TensorDataset, DataLoader


class BCModel(nn.Module):
    def __init__(self,input_dim, latent_dim,output_dim):
        super().__init__()
        self.couche1=nn.Linear(input_dim,latent_dim)
        self.couche2=nn.Linear(latent_dim,latent_dim)
        self.couche3=nn.Linear(latent_dim, output_dim)
                                  
    def forward(self,x):
        y = F.relu(self.couche1(x))
        y = F.relu(self.couche2(y))
        return torch.tanh(self.couche3(y))

        


if __name__=="__main__":

    #on recupère les donné
    data = torch.load("datasets/bc_dataset.pt")
    observations = data["obs"]      
    actions = data["actions"] 
    #normaliser
    obs_mean = observations.mean(0)
    obs_std = observations.std(0) + 1e-8  
    observations = (observations - obs_mean) / obs_std
    
    # Convert en float32
    observations = torch.tensor(observations, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)


    #le modele
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCModel(48,256,2).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    mse = nn.MSELoss()


    #datatset et dataLoader
    train_dataset = TensorDataset(observations, actions)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    EPOCHS=1000
    #boucle d'app
    losses=[]
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss=0

        for x,y in train_loader:
            optim.zero_grad()
            x,y =x.to(device),y.to(device)
            pred=model(x)
            loss=mse(pred,y)
            epoch_loss+=loss
            loss.backward()
            optim.step()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.6f}")

    torch.save(model.state_dict(), "stk_actor/bc_model.pth")

    