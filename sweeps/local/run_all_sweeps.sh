#!/bin/bash
# Run sweep 2 then reliable MLP sweep on GPU 0
# GPU 1 runs sweep 2 independently

echo "$(date) Waiting for sweep 2 GPU0 to finish..."
wait

echo "$(date) Sweep 2 GPU0 done, starting reliable MLP sweep..."
cd ${PROJECT_ROOT:-.}/local_tests
bash sweep1_reliable.sh 0

echo "$(date) All sweeps complete!"
