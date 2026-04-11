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
for modality in ${MODALITY}
do
for batch_size in 64
do
for hidden_dim in 256
do
for tau in 0.5
do
for threshold in 0.4
do
for num_layers_pred in 2
do
for num_layers_fus in 2
do
for num_layers_enc in 2
do
for num_heads in 4
do
for interaction_loss_weight in 0.5
do
for temperature_rw in 2
do
for hidden_dim_rw in 256
do
for num_layer_rw in 3
do
CUDA_VISIBLE_DEVICES="$device" python src/imoe/train_interpretcc.py \
    --data dreamt \
    --dreamt_data_dir "$dreamt_data_dir" \
    --temperature_rw $temperature_rw \
    --hidden_dim_rw $hidden_dim_rw \
    --num_layer_rw $num_layer_rw \
    --train_epochs 50 \
    --modality "$modality" \
    --fusion_sparse False \
    --lr $lr \
    --batch_size $batch_size \
    --hidden_dim $hidden_dim \
    --num_layers_enc $num_layers_enc \
    --num_layers_fus $num_layers_fus \
    --num_layers_pred $num_layers_pred \
    --tau $tau \
    --hard True \
    --threshold $threshold \
    --num_heads $num_heads \
    --dropout 0.5 \
    --n_runs "$N_RUNS" \
    --max_seeds "$SEED_LIMIT" \
    "${RESULTS_LOG_ARGS[@]}" \
    --interaction_loss_weight $interaction_loss_weight \
    --save False \
    --use_common_ids True \
    "${EXTRA_ARGS[@]}"
done
done
done
done
done
done
done
done
done
done
done
done
done
done