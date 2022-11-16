CUDA_VISIBLE_DEVICES=1 \
python -m torch.distributed.launch \
--master_port 29411 \
--nproc_per_node 1 \
--nnodes=1 \
--node_rank=0 \
training/exp_runner.py \
--conf confs/dtu_grids_3views_reg.conf  \
--exps_folder exps_ablation \
--nepoch 5000 \
--scan_id 97