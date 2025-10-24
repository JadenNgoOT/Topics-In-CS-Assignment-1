
# Topics in CS assignment 1 
## Vizdoom

#### Summary
This project trains an AI to play DOOM using Vizdoom which develops the AI by using only visual information.

#### Installation 
For a guide on how to install vizdoom you can go here https://github.com/Farama-Foundation/ViZDoom

#### How to run on WSL/Linux
>./setup.sh (downloads dependencies)
>
>python3 -m venv .venv && source .venv/bin/activate (activates venv)
>
>Training:
>python train.py --scenario [scenario] --reward [reward model] --timesteps [timesteps]
>optionals: train.py arguments: --render [render T/F (1/0)] --seed [000000] --num_envs [1-8]
>
>Evaluation:
>python eval.py --scenario [scenario] --reward [reward model] --episodes [num of episodes] 
>optionals: eval.py arguments: --render [render T/F (1/0)] --seed [000000] --csv_out [/output_path]

#### How to run on Windows using py launcher
>py -m pip install vizdoom gymnasium pygame tensorboard stable-baselines3[extra] (will create the txt folder later)
>
>py -m venv .venv (creates the venv)
>
>Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass (this allows the scripts to run for the python virtual enviroment temporarily)
>
>.venv\Scripts\activate (activates venv)
>
>Training:
>py train.py train.py --scenario [scenario] --reward [reward model] --timesteps [timesteps]
>optionals: train.py arguments: --render [render T/F (1/0)] --seed [000000] --num_envs [1-8]
>
>Evaluation:
>py eval.py --scenario [scenario] --reward [reward model] --episodes [num of episodes] 
>optionals: eval.py arguments: --render [render T/F (1/0)] --seed [000000] --csv_out [/output_path]

#### to run the already trained models on WSL/Linux
>./setup.sh (downloads dependencies)
>
>python3 -m venv .venv && source .venv/bin/activate (activates venv)
>
>python eval.py --scenario defend_the_line --reward kill_fast --episodes 5 --render 1 


##### Notes:
>in .venv/lib/python3.12/site-packages/vizdoom/scenarios replace defend_the_line and deadly_corridor files with the ones provided
