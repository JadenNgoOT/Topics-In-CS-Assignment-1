./setup.sh (downloads dependencies)

python3 -m venv .venv && source .venv/bin/activate (activates venv)

python train.py --timesteps 1000000 (trains model)

python eval.py --/models ./models/ppo_vizdoom_basic --episodes 5 --render 1 (evals model)