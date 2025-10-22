
num_cb=16
n_neighbours=10
min_dist=0.1

# for group in 0; do
#     python analysis/draw_reconstructed_embeddings.py \
#         --input-manifest data_hdf5/vq_wavlm_large_layer_21_normalize_1_libri_cb_${num_cb}/vox1_cuts_test.jsonl.gz \
#         --num-codebooks $num_cb --embed-dim 1024 \
#         --group $group \
#         --quantizer-path data/quantizer/wavlm-large-layer-21-normalize-1-libri-cb-$num_cb.pt \
#         --suffix wavlm-mvq-cb-$num_cb-group-$group-n-neighours-$n_neighbours-min-dist-$min_dist-pre-defined-group-norm \
#         --n-neighbours $n_neighbours --min-dist $min_dist
# done


for group in -1; do
    python analysis/draw_reconstructed_embeddings.py \
        --input-manifest data_hdf5/vq_wavlm_large_layer_21_normalize_1_libri_cb_${num_cb}/librispeech_cuts_dev-clean.jsonl.gz \
        --num-codebooks $num_cb --embed-dim 1024 \
        --group $group \
        --quantizer-path data/quantizer/wavlm-large-layer-21-normalize-1-libri-cb-$num_cb.pt \
        --suffix ls-dev-clean-wavlm-mvq-cb-$num_cb-group-$group-n-neighours-$n_neighbours-min-dist-$min_dist-pre-defined-group \
        --n-neighbours $n_neighbours --min-dist $min_dist
done