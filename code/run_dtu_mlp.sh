CUDA_VISIBLE_DEVICES=3 \
python -m torch.distributed.launch \
--master_port 29513 \
--nproc_per_node 1 \
--nnodes=1 \
--node_rank=0 \
training/exp_runner.py \
--conf confs/dtu_mlp_3views_reg.conf  \
--nepoch 20000 \
--scan_id 65