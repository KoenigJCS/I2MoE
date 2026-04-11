device="${device:-0}"
dreamt_data_dir="${dreamt_data_dir:-data/dreamt}"
N_RUNS="${N_RUNS:-3}"
SEED_LIMIT="${SEED_LIMIT:-0}"
RESULTS_LOG="${RESULTS_LOG:-}"
MODALITY="${MODALITY:-ALL}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
case "$1" in
--device)
device="$2"
shift 2
;;
--dreamt_data_dir|--dreamt-data-dir)
dreamt_data_dir="$2"
shift 2
;;
--modality|--modalities)
MODALITY="$2"
shift 2
;;
--n_runs|--n-runs)
N_RUNS="$2"
shift 2
;;
--max_seeds|--max-seeds|--seed_limit|--seed-limit)
SEED_LIMIT="$2"
shift 2
;;
--results_log|--results-log)
RESULTS_LOG="$2"
shift 2
;;
--)
shift
EXTRA_ARGS+=("$@")
break
;;
*)
EXTRA_ARGS+=("$1")
shift
;;
esac
done

RESULTS_LOG_ARGS=()
if [[ -n "$RESULTS_LOG" ]]; then
RESULTS_LOG_ARGS=(--results_log "$RESULTS_LOG")
fi

for lr in 1e-4
do
for temperature_rw in 2
do
for hidden_dim_rw in 256
do
for num_layer_rw in 3
do
for interaction_loss_weight in 0.1 0.3 0.5
do
CUDA_VISIBLE_DEVICES="$device" python src/imoe/train_transformer.py \
    --temperature_rw $temperature_rw \
    --hidden_dim_rw $hidden_dim_rw \
    --num_layer_rw $num_layer_rw \
    --interaction_loss_weight $interaction_loss_weight \
    --lr $lr \
    --data dreamt \
    --dreamt_data_dir "$dreamt_data_dir" \
    --gate None \
    --train_epochs 50 \
    --modality "$MODALITY" \
    --fusion_sparse False \
    --batch_size 32 \
    --hidden_dim 256 \
    --num_layers_fus 2 \
    --num_layers_enc 2 \
    --num_layers_pred 2 \
    --num_patches 8 \
    --num_experts 4 \
    --num_routers 1 \
    --top_k 2 \
    --num_heads 4 \
    --dropout 0.5 \
    --n_runs "$N_RUNS" \
    --max_seeds "$SEED_LIMIT" \
    "${RESULTS_LOG_ARGS[@]}" \
    --gate_loss_weight 0.01 \
    --save False \
    --use_common_ids True \
    "${EXTRA_ARGS[@]}"
done
done
done
done
done