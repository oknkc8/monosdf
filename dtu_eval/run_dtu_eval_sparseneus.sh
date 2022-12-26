CUDA_VISIBLE_DEVICES=1 \
python evaluate_single_scene.py \
--input_mesh ../sparseneus_exp/mesh_00000000_scan122_000057.png_lod0.ply \
--scan_id 122 \
--output_dir sparseneus/dtu_scn122 \
--sparseneus True