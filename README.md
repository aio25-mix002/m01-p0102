# README

## Setups
### Create environments
```
conda create -n aio25mix002-m01p0102-rag python=3.11 
conda activate aio25mix002-m01p0102-rag
```
to delete an existing environment
```
conda deactivate aio25mix002-m01p0102-rag
conda env remove -n aio25mix002-m01p0102-rag
```

### Install packages
```
pip install -r requirements-torch.txt
pip install -r requirements.txt
```

## Experiment
### Conversation Memory
```
User: Chào bạn, mình tên MIX002 là nhóm học tập về AI
Assistant: Chào bạn MIX002! Tôi là Assistant của AI VIETNAM
User: MIX002 là ai?
Assistant: MIX002 là nhóm học tập về AI tại AI VIETNAM
```