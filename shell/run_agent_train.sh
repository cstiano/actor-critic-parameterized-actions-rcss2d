#!/bin/bash

# Configure the Python Path to use in the project
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH=$PYTHONPATH:$DIR/..

# Run HFO
HFO/bin/HFO --fullstate --no-logging --untouched-time=$3 --headless --offense-agents=1 --defense-npcs=1 --offense-team=helios --defense-team=helios --trials $2 &
sleep 5
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)
python ./src/agent/$1.py &
echo "conectar"
sleep 4

trap "kill -TERM -$$" SIGINT
wait