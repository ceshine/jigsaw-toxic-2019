rm toxic.zip
7za a -bd -mx=0 toxic.zip toxic/*.py setup.py
rm helperbot.zip
7za a -bd -mx=0 /home/ceshine/kaggle-in-progress/toxic2019/helperbot.zip pytorch_helper_bot/*.py pytorch_helper_bot/helperbot/*.py