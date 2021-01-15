#!/bin/sh
export PYTHONPATH=/home/cthota/machine_learning/workspaces/git_project_bda:$PYTHONPATH
nohup python3 /home/cthota/machine_learning/workspaces/git_project_bda/apps/classification_liwc.py > $1 &
echo "script started"
