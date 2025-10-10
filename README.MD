
# Topics in CS assignment 1 
## Vizdoom

#### Summary
This project trains an AI to play DOOM using Vizdoom which develops the AI by using only visual information.

#### Installation 
For a guide on how to install vizdoom you can go here https://github.com/Farama-Foundation/ViZDoom

#### How to run on WSL
>./setup.sh (downloads dependencies)
>
>python3 -m venv .venv && source .venv/bin/activate (activates venv)
>
>python train.py --scenario [scenario] --reward [reward model] --timesteps [timesteps]
>
>python eval.py --/models ./models/[model name] --episodes [num of episodes] --render [render T/F (0/1)]

#### How to run on Windows using py launcher
>py -m pip install vizdoom gymnasium pygame tensorboard stable-baselines3[extra] (will create the txt folder later)
>
>py -m venv .venv (creates the venv)
>
>Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass (this allows the scripts to run for the python virtual enviroment temporarily)
>
>.venv\Scripts\activate (activates venv)
>
>py train.py --scenario [scenario] --reward [reward model] --timesteps [timesteps] (trains the model)
>
>py eval.py --models ./models/ppo_vizdoom_basic --episodes 5 --render 1 (evaluates the model)
