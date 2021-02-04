#!/bin/sh
export PYTHONPATH=/DATA/workspaces/project_BDA/:$PYTHONPATH
nohup python3 /DATA/workspaces/project_BDA/apps/prob_classification.py > $1 &
echo "script started"
