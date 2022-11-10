CUDA_VISIBLE_DEVICES=3 \
python -m torch.distributed.launch \
--master_port 29493 \
--nproc_per_node 1 \
--nnodes=1 \
--node_rank=0 \
training/exp_runner.py \
--conf confs/dtu_grids_3views_reg.conf  \
--scan_id 24