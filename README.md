# TensorForce Bitcoin Trading Bot

A TensorForce-based Bitcoin trading bot (algo-trader). Uses deep reinforcement learning to automatically buy/sell/hold BTC based on price history.

## 1. Setup
- Python 3.6+ (I use template strings a lot)
- Install & setup Postgres
  - Create two databases: `btc_history` and `hyper_runs`. You can call these whatever you want, and just use one db instead of two if you prefer (see Data section).
  - `cp config.example.json config.json`, pop ^ into `config.json`
- Install [TA-Lib](https://github.com/mrjbq7/ta-lib) manually.
- `pip install -r requirements.txt`
  - If issues, try installing these deps manually.
- Install TensorForce from git repo (constantly changing, we chase HEAD)
  - `git clone https://github.com/lefnire/tensorforce.git`
  - `cd tensorforce && pip install -e .`

Note: you'll wanna run this on a GPU rig with some RAM. I'm using a 1080ti and 16GB RAM; 8GB+ is often in used. You _can_ use a standard PC, no GPU (CPU-only); in that case `pip install -I tensorflow==1.5.0rc1` (instead of `tensorflow-gpu`). The only downside is performance; CPU is _way_ slower than GPU for ConvNet computations. Worth evaluating this repo on a CPU before you decide "yeah, it's worth the upgrade."

## 2. Populate Data
- Download [mczielinski/bitcoin-historical-data](https://www.kaggle.com/mczielinski/bitcoin-historical-data)
- Extract to `data/bitcoin-historical-data`
- `python -c 'from data.data import setup_runs_table;setup_runs_table()'`
  - if you get `ModuleNotFoundError: No module named 'data.data'`, prefix commands with `PYTHONPATH=. python ...`
  - If you have trouble with that, just copy/paste the SQL from that file, execute against your `hyper_runs` DB from above.

## 3. Hypersearch
The crux of practical reinforcement learning is finding the right hyper-parameter combo (things like neural-network width/depth, L1 / L2 / Dropout numbers, etc). Some papers have listed optimal default hypers. Eg, the Proximate Policy Optimization (PPO) [paper](https://blog.openai.com/openai-baselines-ppo/) has a set of good defaults. But in my experience, they don't work well for our purposes (time-series / trading). I'll keep my own "best defaults" updated in this project, but YMMV and you'll very likely need to try different hyper combos yourself. The file `hypersearch.py` will search hypers forever, ever honing in on better and better combos (using Bayesian Optimization (BO), see `gp.py`). See Hypersearch section below for more details.

`python hypersearch.py`

Optional flags:
- `--guess <int>`: sometimes you don't want BO, which is pretty willy-nilly at first, to do the searching. Instead you want to try a hunch or two of your own first. See instructions in `utils.py#guess_overrides`.
- `--net-type <lstm|conv2d>`: see discussion below (LSTM v CNN)
- `--boost`: you can optionally use gradient boosting when searching for the best hyper combo, instead of BO. BO is more exploratory and thorough, gradient boosting is more "find the best solution _now_". I tend to use `--boost` after say 100 runs are in the database, since BO may still be dilly-dallying till 200-300 and daylight's burning. Boost will suck in the early runs.
- `--autoencode`: many of you might hit some GPU RAM constraints (hypersearch crashes due to maxed memory). If so, use this flag. It dimensionality-reduces the price-history timesteps so more can fit into RAM.
- `--n-steps <int>`, `--n-tests <int>`: vary how long to train and how often to report. `n-steps` is number of timesteps to train (in 10k; ie `--n-steps 100` means 1M). `n-tests` is how many times to split that and report back to you / save an entry for viz.

## 4. Run
Once you've found a good hyper combo from above (this could take days or weeks!), it's time to run your results.

`python run.py --name <str>`

- `--name <str>` (required): name of the folder to save your run (during training) or load from (during `--live/--test-live`.
- `--id <int>`: the id of some winning hyper-combo you want to run with. Without this, it'll run from the hard-coded hyper defaults.
- `--early-stop <int>`: sometimes your models can overfit. In particular, PPO can give you great performance for a long time and then crash-and-burn. That kind of behavior will be obvious in your visualization (below), so you can tell your run to stop after x consecutive positive episodes (depends on the agent - some find an optimum and roll for 3 positive episodes, some 8, just eyeball your graph).
- `--live`: whooa boy, time to put your agent on GDAX and make real trades! I'm gonna let you figure out how to plug it in on your own, 'cause that's danger territory. I ain't responsible for shit. In fact, let's make that real - disclaimer at the end of README.
- `--test-live`: same as `live`, but without making the real trades. This will start monitoring a live-updated database (from config.json), same as `live`, but instead of making the actual trade, it pretends it did and reports back how much you would have made/lost. Dry-run. You'll definitely want to run this once or twice before running `--live`.

First, run `python run.py [--id 10] --name test`. This will train your model using run 10 (from `hypersearch.py`) and save to `saves/test`. Without `--id` it will use the hard-coded deafults. You can hit `Ctrl-C` _once_ during training to kill training (in case you see a sweet-spot and don't want to overfit).
Second, run `python run.py [--id 10] --name test --[test-]live` to run in live/test-live mode. If you used `--id` before, use it again here so that loading the model matches it to its net architecture.

## 5. Visualize
TensorForce comes pre-built with reward visualization on a TensorBoard. Check out their Github, you'll see. I needed much more customization than that for viz, so we're not using TensorBoard. I created a mini Flask server (2 routes) and a D3 React dashboard where you can slice & dice hyper combos, visualize progression, etc. If you click on a single run, it'll display a graph of the buy/sell signals that agent took in a time-slice (test-set) so you can eyeball whether he's being smart.

- Server: `cd visualize;FLASK_APP=server.py flask run`
- Client:
  - `cd visualize/client`
  - `npm install;npm install -g webpack-dev-server`
  - `npm start` => localhost:8080

![Screen Shot 2021-03-23 at 20 07 04](https://user-images.githubusercontent.com/81108192/112211044-5861be00-8c13-11eb-83ad-3cb5ab026c80.png)

---

## About

This project is a TensorForce-based Bitcoin trading bot (algo-trader). It uses deep reinforcement learning to automatically buy/sell/hold BTC based on what it learns about BTC price history. The bot says "the price will go up next", but it doesn't tell you what to do. 

## Data

So here's how this project splits up databases (see `config.json`). We start with a `history` DB, which has all the historical BTC prices for multiple exchanges. Import it, train on it. Then we have an optionally separate `runs` database, which saves the results of each of your `hypersearch.py` runs. This data is used by our BO or Boost algo to search for better hyper combos. You can have `runs` table in your `history` database if you want, one-and-the-same. I have them separate because I want the `history` DB on localhost for performance reason (it's a major perf difference, you'll see), and `runs` as a public hosted DB, which allows me to collect runs from separate AWS p3.8xlarge running instances.

Then, when you're ready for live mode, you'll want a `live` database which is real-time, constantly collecting exchange ticker data. `--live` will handle keeping up with that database. Again, these can all 3 be the same database if you want, I'm just doing it my way for performance.

## LSTM v CNN
You'll notice the `--net-type <lstm|conv2d>` flag in `hypersearch.py` and `run.py`. This will select between an LSTM Recurrent Neural Networks (RNNs) or Convolutional Neural Networks (CNN). LSTMs work well in NLP which has some maximum 50-word sentences or so. LSTMs mitigated vanilla RNN's vanishing/exploding gradient for such sentences, true - but BTC history is infinite (on-going). Maybe LSTM can only go so far with time-series. Another possibility is that Deep Reinforcement Learning is most commonly researched, published, and open-sourced using CNNs. This because RL is super video-game centric, self-driving cars, all the vision stuff. So maybe the math behind these models lends better to CNNs? Who knows. The point is - experiment with both. Report back on Github your own findings.

So how does CNN even make sense for time-series? Well we construct an "image" of a time-slice, where the x-axis is time (obviously), the y-axis (height) is nothing... it's. The z-axis (channels) is features (OHLCV, VWAP, bid/ask, etc).

## Reinforcement Models

TensorForce has all sorts of models you can play with. This project currently only supports Proximate Policy Optimization (PPO), but I encourage y'all to add in other models (esp VPG, TRPO, DDPG, ACKTR, etc) and submit PRs. ACKTR is the current state-of-the-art Policy Gradient model, but not yet available in TensorForce. PPO is the second-most-state-of-the-art, so we're using that. TRPO is 3rd, VPG is old. DDPG I haven't put much thought into.

Those are the Policy Gradient models. Then there's the Q-Learning approaches (DQNs, etc). We're not using those because they only support discrete actions, not continuous actions. Our agent has one discrete action (buy|sell|hold), and one continuous action (how much?). Without that "how much" continuous flexibility, building an algo-trader would be... well, not so cool. You could do something like (discrete action = (buy-$200, sell-$200, hold)), but I dunno man... continuous is slicker.

## Hypersearch

You're likely familiar with _grid search_ and _random search_ when searching for optimial hyperparameters for machine learning models. Grid search searches literally every possible combo - exhaustive, but takes infinity years (especially w/ the number of hypers we work with in this project). Random search throws a dart at random hyper combos over and over, and you just kill it eventually and take the best. Super naive - it works ok for other ML setups, but in RL hypers are the make-or-break; more than model selection. Seriously, I've found L1 / L2 / Dropout selection more consequential than PPO vs DQN, LSTM vs CNN, etc.

That's why we're using Bayesian Optimization (BO). Or sometimes you'll hear Gaussian Processes (GP), the thing you're optimizing with BO. See `gp.py`. BO starts off like random search, since it doesn't have anything to work with; and over time it hones in on the best hyper combo using Bayesian inference. Super meta - use ML to find the best hypers for your ML - but makes sense. Wait, why not use RL to find the best hypers? We could (and I tried), but deep RL takes 10s of thousands of runs before it starts converging; and each run takes some 8hrs. BO converges much quicker. I've also implemented my own flavor of hypersearch via Gradient Boosting (if you use `--boost` during training); more for my own experimentation.

## Disclaimer

By using this code you accept all responsibility for money lost because of this code.

## Contributing

Thanks all for your contributions...
    
![Screen Shot 2021-03-21 at 19 11 59](https://user-images.githubusercontent.com/81108192/111917690-519f4380-8a79-11eb-9d01-de457b1655f6.png)
    
ETH WALLET: 0xA1134858c168568CBE37649D16723eC8F782e0A2

![Screen Shot 2021-03-21 at 21 56 54](https://user-images.githubusercontent.com/81108192/111922186-5b807100-8a90-11eb-8504-a3fc3ae35052.png)

BTC WALLET: 3N928MmFq51kbf6fE3fxJbtggBhcjMAhSQ

