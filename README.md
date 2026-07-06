# Projet PySTK2-gymnasium / BBRL

Ce projet contient l'implémentation d'un agent d'apprentissage par renforcement (RL) pour l'environnement **SuperTuxKart** en utilisant la bibliothèque **BBRL** et **PySTK2-gymnasium**.

L'agent contrôle la direction (steering) du kart de manière continue et utilise des techniques d'initialisation par **Behavioral Cloning (BC)** pour accélérer l'apprentissage avec des algorithmes comme **SAC** (Soft Actor-Critic i.e learn.py et learn) et **PPO** (Proximal Policy Optimization).

---

## Installation

Pour installer les dépendances nécessaires au projet :

```bash
pip install -r requirements.txt
```

---

## Entraînement par RL

Implémentation de 3 options : 

### SAC avec initialisation BC 
Entraîne un agent SAC en initialisant ses poids avec ceux du modèle BC pré-entraîné :
```bash
PYTHONPATH=. python -m stk_actor.learn
```
Les poids finaux optimaux sont sauvegardés dans `stk_actor/pystk_actor.pth`.

### SAC avec initialisation Aléatoire
Entraîne un agent SAC à partir de zéro (sans pré-entraînement BC) :
```bash
PYTHONPATH=. python -m stk_actor.learnbis
```
Les poids finaux sont sauvegardés dans `stk_actor/pystk_actor_alea.pth`.

### PPO avec initialisation BC
Entraîne un agent PPO en initialisant ses poids avec ceux du modèle BC pré-entraîné :
```bash
PYTHONPATH=. python -m stk_actor.learnPPO
```
Les poids finaux sont sauvegardés dans `stk_actor/pystk_actor_ppo_bc.pth`.

---

