#!/usr/bin/env python
# coding: utf-8

# # COPA Demo

# In[1]:


import sys, os
from pathlib import Path

if not "cwd" in globals():
   cwd = Path(os.getcwd())
sys.path.insert(0, str(cwd.parents[0]))

TASK_NAME = "ISEAR"
BERT_MODEL = "bert-base-uncased"

dataloader_config = {
    "batch_size": 16,
    "data_dir": Path(os.getcwd()).parents[0],
    "splits": ["train", "dev"],
    "max_sequence_length": 60,
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

# In[4]:


from dataloaders import get_dataloaders

# Loading primary task data
copa_dataloaders = get_dataloaders(
    task_name=TASK_NAME,
    tokenizer_name=BERT_MODEL,
    **dataloader_config
)


# In[5]:


from superglue_tasks import task_funcs

# Defining task
copa_task = task_funcs[TASK_NAME](BERT_MODEL)


# In[6]:


from snorkel.classification import MultitaskClassifier
from snorkel.classification import Trainer

copa_model = MultitaskClassifier(tasks=[copa_task])
trainer = Trainer(**trainer_config)


# In[7]:


vars(copa_dataloaders[1].dataset)


# In[8]:


# Training on COPA an dsaving model -- takes a long time on CPU!
trainer.fit(copa_model, copa_dataloaders)
# copa_model.save('best_model_COPA_SuperGLUE_valid_accuracy.pth')


# In[9]:


# Alternatively, download and load trained model run ahead of time to save time
# ! wget -nc https://www.dropbox.com/s/c7dv5vgr5lqon61/best_model_COPA_SuperGLUE_valid_accuracy.pth
# copa_model.load('best_model_COPA_SuperGLUE_valid_accuracy.pth')


# In[10]:


#copa_dev_loader


# In[11]:


# Evaluating model
copa_train_loader, copa_dev_loader = copa_dataloaders
copa_score = copa_model.score([copa_dev_loader])
print(copa_score)
#print(f"COPA (from BERT) Accuracy: {copa_score['COPA/SuperGLUE/valid/accuracy']}")
