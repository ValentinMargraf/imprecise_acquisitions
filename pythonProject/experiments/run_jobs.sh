#!/bin/bash

# run_jobs.sh
for i in $(seq 0 9); do
    echo "Launching job $i ..."
    python run_bbob.py $i > logs/job_$i.out 2>&1 &
done

wait
echo "All 20 jobs finished."
