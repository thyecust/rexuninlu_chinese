echo "experiment for test, ${data_name}"

torchrun --nproc_per_node=${nproc} main.py \
        --bert_model_dir=${bert_model_dir} \
        --data_path=${data_path} \
        --run_name=${run_name} \
        --task_metrics=${metrics} \
        --do_predict=True \
        --per_device_train_batch_size=${batch_size} \
        --gradient_accumulation_steps=${grad_acc} \
        --per_device_eval_batch_size=${batch_size} \
        --evaluation_strategy=no \
        --num_train_epochs=${epochs} \
        --learning_rate=${lr} \
        --lr_scheduler_type=${lr_type} \
        --log_level=info \
        --logging_strategy=epoch \
        --logging_steps=${logging_steps} \
        --seed=42 \
        --fp16 \ #  --no_cuda=True when only cpu available \
        --report_to=none \
        --save_strategy=epoch \
        --save_total_limit=3 \
        --greater_is_better=True \
        --metric_for_best_model=f1 \
        --verbose_debug \
        --remove_unused_columns=False \
        --in_low_res=true \
        --output_dir=${output_dir} \
        --load_checkpoint=${load_checkpoint}