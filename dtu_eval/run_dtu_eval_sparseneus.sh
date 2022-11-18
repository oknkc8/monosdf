CUDA_VISIBLE_DEVICES=1 \
python evaluate_single_scene.py \
--input_mesh ../sparseneus_exp/mesh_00012000_scan114_000013.png_lod0.ply \
--scan_id 114 \
--output_dir sparseneus/dtu_scn114 \
--sparseneus True