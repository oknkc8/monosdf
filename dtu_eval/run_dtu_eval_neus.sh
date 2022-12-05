CUDA_VISIBLE_DEVICES=0 \
python evaluate_single_scene.py \
--input_mesh ../neus_exp/00050000_65.ply \
--scan_id 65 \
--output_dir neus/dtu_scn65 \
--sparseneus True