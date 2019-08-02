#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

sys.path.append("..")


# In[2]:


from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader

from toxic.inference_bert import get_token_ids
from toxic.dataset import AUX_COLUMNS, ToxicDataset, collate_examples, SortSampler
from toxic.common import ToxicBot
from toxic.metric import ToxicMetric


# In[3]:


MODEL_PATH = Path("../data/cache/")
DEVICE = "cuda:0"


# In[4]:


tokenizer = joblib.load(str(MODEL_PATH / "bert-base-uncased_tokenizer.jbl"))
model = torch.load(str(MODEL_PATH / "bert-base-uncased_-1_yuval_220_f0.pth")).to(DEVICE)


# In[5]:


collate_fn = partial(
    collate_examples,
    truncate_len=220,
    pad=0,
    closing_id=tokenizer.vocab["[SEP]"],
    mode="both"
)


# ![](https://pbs.twimg.com/media/DICFy_jWsAE6s6V?format=jpg&name=small)
# [source](https://twitter.com/jessamyn/status/900867154412699649)

# In[6]:


test_text = [
    "I am a man",
    "I am a woman",
    "I am a lesbian",
    "I am gay man",
    "I am dyke",
    "I am a white man",
    "I am a gay woman",
    "I am a white woman",
    "I am a gay white man",
    "I am a black man",
    "I am a gay white man",
    "I am a gay black man",
    "I am a black woman",
    "I am a gay black woman"
]
df = pd.DataFrame(dict(comment_text=test_text))


# In[7]:


tokens = get_token_ids(
    df, tokenizer, is_bert=True)
test_ds = ToxicDataset(df, tokens, labeled=False)
test_loader = DataLoader(
    test_ds,
    collate_fn=collate_fn,
    batch_size=32,
    num_workers=0,
    pin_memory=True
)


# In[8]:


with torch.no_grad():
    results = []
    for batch, _ in test_loader:
        results.append(model(batch.cuda()))
    results = torch.sigmoid(torch.cat(results)) * 100
    results.size()


# In[9]:


predictions = pd.DataFrame(results.cpu().numpy(), columns=AUX_COLUMNS)
predictions["text"] = df["comment_text"].values
predictions.shape


# In[10]:


pd.set_option('display.float_format', lambda x: '%.2f' % x)


# In[11]:


predictions.columns


# In[12]:


predictions[["text", "target", "identity_attack", "female", "homosexual_gay_or_lesbian", "black", "white"]].rename(
    columns={"target": "toxic", "homosexual_gay_or_lesbian":"homosexual"})


# ## Other random examples

# In[23]:


test_text = [
    "Pretty sad really. Thousands come out to protest Trump who's in another country......and all those NDP and Green members - who were the driving force behind Proportional Representation - couldn't find the time to get out to support their own cause. That said, you can bet that Trudeau has alredy lost a good portion of that NDP/Green vote. The guy's a liar. Period.",
    "Idiots.",
    "Then put them in jail until they decide to cooperate.  This is ridiculous....",
    "\"He’s a proven liar. Why would you believe anything he has to say?\"\n\nExactly.\nThat\'s why I do not listen to a word Donnie says.",
    "Fascinating young woman, impressive work. It reminds me of one of the reasons I love Alaska so much —\xa0the people Alaska attracts are often just as extraordinary as the landscape itself. Great article, thank you.",
    "Well, at least the Russians are white, for heaven's sakes.  I'd rather have to live next to a nice white drunk Russian than a black Muslim Kenyan whose children's names are Satanic anagrams.",
    "Was I posted yesterday, it is interesting to note that under Session's watch only three black people have been appointed in Alabama for the federal courts. This despite the fact that black people make up over 39% of the population of that state. What underlines this reality must be Session's unconscious, if not conscious, attitude towards blacks in general."
]
df = pd.DataFrame(dict(comment_text=test_text))


# In[24]:


tokens = get_token_ids(
    df, tokenizer, is_bert=True)
print([len(x) for x in tokens])
test_ds = ToxicDataset(df, tokens, labeled=False)
test_loader = DataLoader(
    test_ds,
    collate_fn=collate_fn,
    batch_size=32,
    num_workers=0,
    pin_memory=True
)
with torch.no_grad():
    results = []
    for batch, _ in test_loader:
        results.append(model(batch.cuda()))
    results = torch.sigmoid(torch.cat(results)) * 100
    results.size()
predictions = pd.DataFrame(results.cpu().numpy(), columns=AUX_COLUMNS)
predictions["text"] = df["comment_text"].values
predictions[["text", "target", "identity_attack", "female", "homosexual_gay_or_lesbian", "black", "white"]].rename(
    columns={"target": "toxic", "homosexual_gay_or_lesbian":"homosexual"})


# ## Validate
# Make sure the mode is set up correctly.

# In[80]:


df_valid, tokens_valid = joblib.load(str(MODEL_PATH / "valid_bert-base-uncased_-1_yuval_f0.jbl"))
idx = np.random.choice(np.arange(df_valid.shape[0]), 32 * 1000)
df_valid, tokens_valid = df_valid.iloc[idx].reset_index(drop=True), tokens_valid[idx]
valid_ds = ToxicDataset(df_valid, tokens_valid, labeled=True)
val_sampler = SortSampler(valid_ds, key=lambda x: len(valid_ds.tokens[x]))
df_valid = df_valid.iloc[list(iter(val_sampler))]
print(df_valid.target.describe())


# In[81]:


valid_loader = DataLoader(
    valid_ds,
    collate_fn=collate_fn,
    batch_size=64,
    num_workers=0,
    pin_memory=True,
    sampler=val_sampler
)


# In[82]:


bot = ToxicBot(
    checkpoint_dir=Path("/tmp/"),
    log_dir=Path("/tmp/"),
    model=model, train_loader=None,
    val_loader=None, optimizer=None,
    echo=False,
    criterion=None,
    avg_window=100,
    callbacks=[],
    pbar=False,
    use_tensorboard=False,
    device=DEVICE
)
valid_pred, valid_y = bot.predict(valid_loader, return_y=True)


# In[84]:


pd.set_option('precision', 4)
metric = ToxicMetric(df_valid)
metric(valid_y, valid_pred)


# In[ ]:




