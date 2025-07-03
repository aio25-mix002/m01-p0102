# README
- [README](#readme)
  - [Setups](#setups)
    - [Create environments](#create-environments)
    - [Install packages](#install-packages)
    - [Environment variables](#environment-variables)
  - [Run application](#run-application)
    - [Via CLI](#via-cli)
    - [Via VS Code Launch Profile](#via-vs-code-launch-profile)
    - [Via Google Colab](#via-google-colab)
  - [Experiment](#experiment)
    - [RAG flow](#rag-flow)
    - [Conversation Memory](#conversation-memory)


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


### Environment variables
We can create an `.env.local` to store environment variables of the application. 
An example of variables can be found in `.env.example`. 

## Run application
### Via CLI
**Within quantization**
```
streamlit run ./rag_chatbot.py
```

**Without quantization**
```
streamlit run ./rag_chatbot.py -- --no-quantization
```
- The `--` is needed to separate Streamlit's arguments from your script's arguments.

### Via VS Code Launch Profile
We can also start the app via the launch profile.
- On windows: Debug Streamlit App (Miniconda - Windows)

### Via Google Colab
We can run this app via Google Colab, checkout `runbook_m01p0102.ipynb` to learn more.

## Experiment
### RAG flow
We can use the sample file from `./examples/YOLOv10_Tutorials.pdf`. 
Then ask some questions like: 
- YOLO là gì?
- Object Detection là gì?
- ...
 
### Conversation Memory
```
User: Chào bạn, mình tên MIX002 là nhóm học tập về AI
Assistant: Chào bạn MIX002! Tôi là Assistant của AI VIETNAM
User: MIX002 là ai?
Assistant: MIX002 là nhóm học tập về AI tại AI VIETNAM
```