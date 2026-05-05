#!/usr/bin/env bash
# script to run all experiments sequentially 

echo "new test started" $date >> /export/scratch2/ima/MultiFIX_GECCO25_code/hpo_results/commands_pid_$$.log

echo $!

export TRAINING='ft_comp_comp'
python -u scripts/hpo.py > /export/scratch2/ima/MultiFIX_GECCO25_code/hpo_results/train_fusion_comp_comp_$$.log 2>&1 && \
echo $! $TRAINING "python -u scripts/hpo.py > /export/scratch2/ima/MultiFIX_GECCO25_code/hpo_results/train_fusion_comp_comp_$$.log 2>&1 &" >> /export/scratch2/ima/MultiFIX_GECCO25_code/hpo_results/commands_pid_$$.log

export TRAINING='ft_comp_part'
python -u scripts/hpo.py > /export/scratch2/ima/MultiFIX_GECCO25_code/hpo_results/train_fusion_comp_part_$$.log 2>&1 && \
echo $! $TRAINING "python -u scripts/hpo.py > /export/scratch2/ima/MultiFIX_GECCO25_code/hpo_results/train_fusion_comp_part_$$.log 2>&1 &" >> /export/scratch2/ima/MultiFIX_GECCO25_code/hpo_results/commands_pid_$$.log


export TRAINING='ft_comp_none'
python -u scripts/hpo.py > /export/scratch2/ima/MultiFIX_GECCO25_code/hpo_results/train_fusion_comp_none_$$.log 2>&1 && \
echo $! $TRAINING >> /export/scratch2/ima/MultiFIX_GECCO25_code/hpo_results/commands_pid_$$.log

export TRAINING='ft_part_part'
python -u scripts/hpo.py > /export/scratch2/ima/MultiFIX_GECCO25_code/hpo_results/train_fusion_part_part_$$.log 2>&1 && \
echo $! $TRAINING >> /export/scratch2/ima/MultiFIX_GECCO25_code/hpo_results/commands_pid_$$.log

export TRAINING='ft_part_none'
python -u scripts/hpo.py > /export/scratch2/ima/MultiFIX_GECCO25_code/hpo_results/train_fusion_part_none_$$.log 2>&1 && \
echo $! $TRAINING >> /export/scratch2/ima/MultiFIX_GECCO25_code/hpo_results/commands_pid_$$.log

export TRAINING='ft_none_none'
python -u scripts/hpo.py > /export/scratch2/ima/MultiFIX_GECCO25_code/hpo_results/train_fusion_none_none_$$.log 2>&1 && \
echo $! $TRAINING >> /export/scratch2/ima/MultiFIX_GECCO25_code/hpo_results/commands_pid_$$.log


# adding for unimodal img and unimodal tab
# nohup python -u scripts/hpo.py > /export/scratch2/ima/MultiFIX_GECCO25_code/hpo_results/train_img_$$.log 2>&1 &
# nohup python -u scripts/hpo.py > /export/scratch2/ima/MultiFIX_GECCO25_code/hpo_results/train_tab_$$.log 2>&1 &
