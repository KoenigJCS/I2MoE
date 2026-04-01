export device=0
export dreamt_data_dir=data/dreamt

modality_arg=BXYZDTHIJKLMNPQCERFUVWGAO
dreamt_max_files=0
dreamt_use_class_weights=True
n_runs=3
train_epochs=50
lr=1e-4

while [[ $# -gt 0 ]]; do
case "$1" in
    --modality|--modalities)
        modality_arg="$2"
        shift 2
        ;;
    --max-files|--dreamt-max-files)
        dreamt_max_files="$2"
        shift 2
        ;;
    --use-class-weights|--dreamt-use-class-weights)
        dreamt_use_class_weights="$2"
        shift 2
        ;;
    --n-runs)
        n_runs="$2"
        shift 2
        ;;
    --train-epochs)
        train_epochs="$2"
        shift 2
        ;;
    --data-dir|--dreamt-data-dir)
        dreamt_data_dir="$2"
        shift 2
        ;;
    --device)
        device="$2"
        shift 2
        ;;
    --lr)
        lr="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        return 1 2>/dev/null || exit 1
        ;;
esac
done

for lr in $lr
do
for modality in "$modality_arg"
do
for batch_size in 128
do
for hidden_dim in 64
do
for num_patches in 4
do
for num_experts in 8
do
for num_layers_pred in 2
do
for num_layers_fus in 2
do
for num_layers_enc in 2
do
for num_heads in 4
do
for interaction_loss_weight in 0.1
do
for temperature_rw in 2
do
for hidden_dim_rw in 256
do
for num_layer_rw in 2
do
CUDA_VISIBLE_DEVICES=$device python src/imoe/train_moepp.py \
    --temperature_rw $temperature_rw \
    --hidden_dim_rw $hidden_dim_rw \
    --num_layer_rw $num_layer_rw \
    --data dreamt \
    --dreamt_data_dir $dreamt_data_dir \
    --dreamt_max_files $dreamt_max_files \
    --dreamt_use_class_weights $dreamt_use_class_weights \
    --train_epochs $train_epochs \
    --modality $modality \
    --fusion_sparse False \
    --lr $lr \
    --batch_size $batch_size \
    --hidden_dim $hidden_dim \
    --num_layers_enc $num_layers_enc \
    --num_layers_fus $num_layers_fus \
    --num_layers_pred $num_layers_pred \
    --num_patches $num_patches \
    --num_experts $num_experts \
    --num_heads $num_heads \
    --dropout 0.5 \
    --n_runs $n_runs \
    --interaction_loss_weight $interaction_loss_weight \
    --save False \
    --use_common_ids True
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