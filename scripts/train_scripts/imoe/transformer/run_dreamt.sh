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
for temperature_rw in 2
do
for hidden_dim_rw in 256
do
for num_layer_rw in 3
do
for interaction_loss_weight in 0.1 0.3 0.5
do
CUDA_VISIBLE_DEVICES=$device python src/imoe/train_transformer.py \
    --temperature_rw $temperature_rw \
    --hidden_dim_rw $hidden_dim_rw \
    --num_layer_rw $num_layer_rw \
    --interaction_loss_weight $interaction_loss_weight \
    --lr $lr \
    --data dreamt \
    --dreamt_data_dir $dreamt_data_dir \
    --dreamt_max_files $dreamt_max_files \
    --dreamt_use_class_weights $dreamt_use_class_weights \
    --gate None \
    --train_epochs $train_epochs \
    --modality $modality \
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
    --n_runs $n_runs \
    --gate_loss_weight 0.01 \
    --save False \
    --use_common_ids True
done
done
done
done
done
done