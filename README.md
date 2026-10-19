# Progent: Securing AI Agents with Privilege Control
Check out our paper [here](https://arxiv.org/abs/2504.11703).

## Installation
```bash
pip install -e .
```

## Experiments in the paper
### Agentdojo
```bash
cd agentdojo
pip install -e . # install agentdojo
cd ..
pip install -e . # install progent
cd agentdojo
./run.sh
```
Check out more in [agentdojo/README.md](agentdojo/README.md)

### ASB
```bash
cd asb
pip install -r requirements.txt # install asb
cd ..
pip install -e . # install progent
cd asb
python scripts/agent_attack.py --cfg_path config/OPI.yml
```
Check out more in [asb/README.md](asb/README.md)

### Real world agents
```bash
cd agentdojo-mcp
pip install -e . # install agentdojo-mcp
python mcp_server.py # start the mcp server
cd ..
pip install -e . # install progent
cd real-world-agents
pip install -r requirements.txt
./run.sh
```
