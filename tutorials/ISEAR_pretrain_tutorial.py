#!/usr/bin/env python
# coding: utf-8

# # ISEAR Demo

# In[1]:


import sys, os
from pathlib import Path

if not "cwd" in globals():
   cwd = Path(os.getcwd())
sys.path.insert(0, str(cwd.parents[0]))


# In[2]:


TASK_NAME = "ISEAR"
AUX_TASK_NAME = "SWAG"
BERT_MODEL = "bert-base-uncased"

dataloader_config = {
    "batch_size": 18,
    "data_dir": Path(os.getcwd()).parents[0],
    "splits": ["train", "dev"],
    "max_sequence_length": 50,
}

trainer_config = {
    "lr": 2e-4,
    "optimizer": "sgd",
    "n_epochs": 10,
    "checkpointing": 1,
    "logging": 1,
    "grad_clip": None,
}


# ### Train Primary Task from BERT

# In[3]:


from dataloaders import get_dataloaders

# Loading primary task data
isear_dataloaders = get_dataloaders(
    task_name=TASK_NAME,
    tokenizer_name=BERT_MODEL,
    **dataloader_config
)


# In[4]:


from tasks import task_funcs

# Defining task
isear_task = task_funcs[TASK_NAME](BERT_MODEL)


# In[5]:


trainer_config = {
    "lr": 2e-5,
    "optimizer": "adam",
    "n_epochs": 10,
    "checkpointing": 1,
    "logging": 1,
    "l2": 0.001,
}


# In[6]:


from snorkel.classification import MultitaskClassifier
from snorkel.classification import Trainer

isear_model = MultitaskClassifier(tasks=[isear_task])
trainer = Trainer(**trainer_config)


# In[7]:


# Training on ISEAR an dsaving model -- takes a long time on CPU!
trainer.fit(isear_model, isear_dataloaders)
# isear_model.save('best_model_ISEAR_valid_accuracy.pth')


# In[ ]:


# Evaluating model
isear_train_loader, isear_dev_loader = isear_dataloaders
isear_score = isear_model.score([isear_dev_loader])
print(isear_score)

