CUDA_VISIBLE_DEVICES=7 \
python -m torch.distributed.launch \
--master_port 29497 \
--nproc_per_node 1 \
--nnodes=1 \
--node_rank=0 \
training/exp_runner.py \
--conf confs/replica_grids_sparseviews_reg_nowarp.conf  \
--exps_folder exps_ablation_replica_new \
--nepoch 200 \
--scan_id 8