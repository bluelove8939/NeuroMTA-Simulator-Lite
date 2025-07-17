mkdir -p .tmp
python scripts/experiments/multi_core_l2_shared.py > .tmp/multi_core_l2_shared.log
python scripts/experiments/multi_core_l1_no_reuse.py > .tmp/multi_core_l1_no_reuse.log
python scripts/experiments/multi_core_l1_reuse.py > .tmp/multi_core_l1_reuse.log
python scripts/experiments/single_core.py > .tmp/single_core.log