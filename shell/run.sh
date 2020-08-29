#!/bin/bash

# Configure the Python Path to use in the project
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH=$PYTHONPATH:$DIR/..

# Run HFO
HFO/bin/HFO --fullstate --no-logging --headless --offense-agents=1 --defense-npcs=1 --offense-team=$1 --defense-team=$2 --trials $3 &
sleep 5
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)
python ./src/train/base.py &
echo "conectar"
sleep 4

trap "kill -TERM -$$" SIGINT
wait