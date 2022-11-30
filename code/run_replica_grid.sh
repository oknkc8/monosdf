CUDA_VISIBLE_DEVICES=5 \
python -m torch.distributed.launch \
--master_port 29495 \
--nproc_per_node 1 \
--nnodes=1 \
--node_rank=0 \
training/exp_runner.py \
--conf confs/replica_grids_sparseviews_reg.conf  \
--exps_folder exps_ablation_replica \
--nepoch 200 \
--scan_id 2