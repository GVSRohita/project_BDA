#!/bin/sh
export PYTHONPATH=/DATA/workspaces/project_BDA/:$PYTHONPATH
nohup python3 /DATA/workspaces/project_BDA/apps/data_visible.py > $1 &
echo "script started"
