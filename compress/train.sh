# python pre_prepare_data.py --work_dir 'compressLLM_len-510-ratio-15_wo-cmp&pe'
# python ./pre_trainer.py --work_dir 'compressLLM_len-510-ratio-15_wo-cmp&pe' --port 14522
# python ./pre_evaluator.py --work_dir 'compressLLM_len-510-ratio-15_wo-cmp&pe' --batch_size 1


# python instruction_prepare_data.py --work_dir 'compressLLM_len-510-ratio-15_wo-cmp&pe'
# python ./instruction_trainer.py --work_dir 'compressLLM_len-510-ratio-15_wo-cmp&pe' --port 14522
# python ./instruction_evaluator.py --work_dir 'compressLLM_len-510-ratio-15_wo-cmp&pe' --batch_size 1

# python pre_prepare_data.py --work_dir compressLLM_debug
# python ./pre_trainer.py --work_dir compressLLM_debug --port 29500
# python ./pre_evaluator.py --work_dir compressLLM_debug --batch_size 1

# python instruction_prepare_data.py --work_dir compressLLM_debug
# python ./instruction_trainer.py --work_dir compressLLM_debug --port 29500
# python ./instruction_evaluator.py --work_dir compressLLM_debug --batch_size 1


# python vanilla_prepare_data.py --work_dir compressLLM_debug
# python ./vanilla_trainer.py --work_dir compressLLM_debug --port 29500
# python ./vanilla_evaluator.py --work_dir compressLLM_debug --batch_size 1

#python pre_prepare_data.py --work_dir 'compressLLM_baseline_len-510_ratio-5_wo-ae'
#python ./pre_trainer.py --work_dir 'compressLLM_baseline_len-510_ratio-5_wo-ae' --port 14522
#python ./pre_evaluator.py --work_dir 'compressLLM_baseline_len-510_ratio-5_wo-ae' --batch_size 1

#python ./instruction_trainer.py --work_dir 'compressLLM_multi_lora_510_ratio-5_lm' --port 14529
#python ./instruction_evaluator.py --work_dir 'compressLLM_multi_lora_510_ratio-5_lm' --batch_size 1

#python ./pre_trainer.py --work_dir 'compressLLM_multi_lora_510_ratio-5_lm' --port 14528
# bash train.sh

#python ./instruction_trainer.py --work_dir 'compressLLM_multi_lora_510_ratio_lm&cl' --port 14525
#python ./instruction_evaluator.py --work_dir 'compressLLM_multi_lora_510_ratio_lm&cl' --batch_size 1

# 1.重新评估一下baseline的压缩准确率
#python ./pre_evaluator.py --work_dir 'compressLLM_test' --batch_size 1
# 2.评估一下mask用对的准确率
#python ./instruction_evaluator.py --work_dir 'compressLLM_multi_lora_510_ratio-5_lm' --batch_size 1

# 3.对之前的cl_generate产生怀疑，重新评测
#python ./pre_evaluator.py --work_dir 'compressLLM_instruction_baseline_lm' --batch_size 1
#-----------------------------------------------------------------------------------------------------------
# 重新训练一个Multi-Lora的baseline
# 重新训练一个正常的adapter
#python ./pre_trainer.py --work_dir 'compressLLM_baseline_len-510_ratio-5_wo-ae' --port 14522
#python ./pre_evaluator.py --work_dir 'compressLLM_baseline_len-510_ratio-5_wo-ae' --batch_size 1
# 下面三个任务：已经配置好了直接启动
# 任务1：训练完以后再进行微调，此时需要修改config.json 去掉compress_loss
#python ./instruction_trainer.py --work_dir 'compressLLM_instruction_baseline_lm' --port 14523
#python ./instruction_evaluator.py --work_dir 'compressLLM_instruction_baseline_lm' --batch_size 1

# 任务6：同时完成，去掉compress_loss，然后使用Multi-LoRA
#python ./instruction_trainer.py --work_dir 'compressLLM_multi_lora_510_ratio-5_lm' --port 14524
#python ./instruction_evaluator.py --work_dir 'compressLLM_multi_lora_510_ratio-5_lm' --batch_size 1

## 任务7：
#python ./instruction_trainer.py --work_dir 'compressLLM_multi_lora_510_ratio_lm&cl' --port 14525
#python ./instruction_evaluator.py --work_dir 'compressLLM_multi_lora_510_ratio_lm&cl' --batch_size 1
#
## 任务2：
#python ./instruction_trainer.py --work_dir 'compressLLM_instruction_baseline_cl&lm_1-1' --port 14528
#python ./instruction_evaluator.py --work_dir 'compressLLM_instruction_baseline_cl&lm_1-1' --batch_size 1

#-----------------------------------------------------------------------------------------------------------
# 任务5：需要修改一下秩的大小512/256 然后直接启动(这个需要单独来，不然容易爆内存)
#python ./pre_trainer.py --work_dir 'compressLLM_baseline_merge_lora_rank-512_len-510_ratio-5_wo-ae' --port 14526
#python ./pre_evaluator.py --work_dir 'compressLLM_baseline_merge_lora_rank-512_len-510_ratio-5_wo-ae' --batch_size 1
# 修改秩的大小，微调其实不用改，记得把model的修改回去
#python ./instruction_trainer.py --work_dir 'compressLLM_test' --port 14527
#python ./instruction_evaluator.py --work_dir 'compressLLM_test' --batch_size 1
#python ./instruction_trainer.py --work_dir 'compressLLM_test_pro' --port 14528
#-----------------------------------------------------------------------------------------------------------

# 预训练任务5
#python ./pre_trainer.py --work_dir 'compressLLM_pwc_continue_pretrain' --port 14529
#python ./pre_evaluator.py --work_dir 'compressLLM_pwc_continue_pretrain' --batch_size 1
# todo:compressLLM_test是只用了compress效果好的，目前还没测试
#python ./instruction_trainer.py --work_dir 'compressLLM_test' --port 14527
#python ./instruction_evaluator.py --work_dir 'compressLLM_test' --batch_size 1
# ---------------------------------------11.16-----------------------------------------
# todo: 11.compressLLM_random_instruction_rank-512_lm&cl
#python ./instruction_trainer.py --work_dir 'compressLLM_random_instruction_rank-512_lm&cl' --port 14521
#python ./instruction_evaluator.py --work_dir 'compressLLM_random_instruction_rank-512_lm&cl' --batch_size 1
# 正常微调：8000M、多LoRA微调：10000

# todo: 10.compressLLM_random_instruction_rank-512_lm
#python ./instruction_evaluator.py --work_dir 'compressLLM_random_instruction_rank-512_lm' --batch_size 1

# todo: 12.compressLLM_random_instruction_rank-512_multi-lora_lm&cl
#python ./instruction_trainer.py --work_dir 'compressLLM_random_instruction_rank-512_multi-lora_lm&cl' --port 14522
#python ./instruction_evaluator.py --work_dir 'compressLLM_random_instruction_rank-512_multi-lora_lm&cl' --batch_size 1

# todo: 9.compressLLM_test是只用了compress效果好的，目前还没测试
#python ./instruction_evaluator.py --work_dir 'compressLLM_test' --batch_size 1

# -------------------------------------11.17-------------------------------------------------------

# todo: 16.compressLLM_random_instruction_multi-lora_lm&cl
#python ./instruction_trainer.py --work_dir 'compressLLM_random_instruction_multi-lora_lm&cl' --port 14522
#python ./instruction_evaluator.py --work_dir 'compressLLM_random_instruction_multi-lora_lm&cl' --batch_size 1
# todo: 14.compressLLM_random_instruction_lm
#python ./instruction_trainer.py --work_dir 'compressLLM_random_instruction_lm' --port 14522
#python ./instruction_evaluator.py --work_dir 'compressLLM_random_instruction_lm' --batch_size 1
# todo: 15.compressLLM_random_instruction_lm&cl
#python ./instruction_trainer.py --work_dir 'compressLLM_random_instruction_lm&cl' --port 14522
#python ./instruction_evaluator.py --work_dir 'compressLLM_random_instruction_lm&cl' --batch_size 1
# todo: 17.compressLLM_random_instruction_(pre-train-multi-lora)_multi-lora_lm&cl
#python ./instruction_trainer.py --work_dir 'compressLLM_random_instruction_(pre-train-multi-lora)_multi-lora_lm&cl' --port 14522
#python ./instruction_evaluator.py --work_dir 'compressLLM_random_instruction_(pre-train-multi-lora)_multi-lora_lm&cl' --batch_size 1

# -------------------------------------11.18-------------------------------------------------------

# todo: 18.compressLLM_test_pro：compress_token=0猜想，看下游任务是否有变化
#python ./instruction_trainer.py --work_dir 'compressLLM_test_pro' --port 14523
#python ./instruction_evaluator.py --work_dir 'compressLLM_test_pro' --batch_size 1
# todo: 18.compressLLM_test：compress_token=0猜想，看下游任务是否有变化
#python ./instruction_trainer.py --work_dir 'compressLLM_test' --port 14526
#python ./instruction_evaluator.py --work_dir 'compressLLM_test' --batch_size 1

# -------------------------------------11.19-------------------------------------------------------
# todo: 7. 新宇学长说的pwc继续预训练
#python ./pre_trainer.py --work_dir 'compressLLM_pwc_continue_pretrain' --port 14529
#python ./pre_evaluator.py --work_dir 'compressLLM_pwc_continue_pretrain' --batch_size 1

# todo: 18.compressLLM_random_instruction_(pre-train-multi-lora)_multi-lora_lm
#python ./instruction_trainer.py --work_dir 'compressLLM_random_instruction_(pre-train-multi-lora)_multi-lora_lm' --port 14526
#python ./instruction_evaluator.py --work_dir 'compressLLM_random_instruction_(pre-train-multi-lora)_multi-lora_lm' --batch_size 1

# todo: 19.compressLLM_random_instruction_cut_gradient_multi-lora_lm
#python ./instruction_trainer.py --work_dir 'compressLLM_random_instruction_cut_gradient_multi-lora_lm' --port 14526
#python ./instruction_evaluator.py --work_dir 'compressLLM_random_instruction_cut_gradient_multi-lora_lm' --batch_size 1

# -------------------------------------11.20-------------------------------------------------------
# todo: 20.compressLLM_baseline_len-510_ratio-5_wo-ae_sl
#python ./pre_trainer.py --work_dir 'compressLLM_baseline_len-510_ratio-5_wo-ae_sl' --port 14529
#python ./pre_evaluator.py --work_dir 'compressLLM_baseline_len-510_ratio-5_wo-ae_sl' --batch_size 1

# todo: 21.compressLLM_test_pro
#python ./instruction_trainer.py --work_dir 'compressLLM_test_pro' --port 14526
#python ./instruction_evaluator.py --work_dir 'compressLLM_test_pro' --batch_size 1

# todo: 21.compressLLM_baseline_only_lm
#python ./pre_trainer.py --work_dir 'compressLLM_baseline_only_lm' --port 14529
#python ./pre_evaluator.py --work_dir 'compressLLM_baseline_only_lm' --batch_size 1
#python ./instruction_trainer.py --work_dir 'compressLLM_baseline_only_lm' --port 14524
#python ./instruction_evaluator.py --work_dir 'compressLLM_baseline_only_lm' --batch_size 1
#python ./instruction_trainer.py --work_dir 'compressLLM_random_instruction_ae' --port 14527
#python ./instruction_evaluator.py --work_dir 'compressLLM_random_instruction_ae' --batch_size 1

# -------------------------------------11.21-------------------------------------------------------
# todo: 21.compressLLM_random_instruction_compress_token_to_context
#python ./instruction_trainer.py --work_dir 'compressLLM_random_instruction_compress_token_to_context' --port 14524
#python ./instruction_evaluator.py --work_dir 'compressLLM_random_instruction_compress_token_to_context' --batch_size 1

# todo: 21.compressLLM_test(KL)
#python ./instruction_trainer.py --work_dir 'compressLLM_test_mmlu_all_18' --port 14527
#python ./instruction_trainer.py --work_dir 'compressLLM_test_rajpurkar_squad_18' --port 14525
#python ./instruction_evaluator.py --work_dir 'compressLLM_test_mmlu_all_18' --batch_size 1
#python ./instruction_evaluator.py --work_dir 'compressLLM_test_rajpurkar_squad_18' --batch_size 1
#python ./instruction_evaluator.py --work_dir 'compressLLM_test_ucinlp_drop_18' --batch_size 1

# python instruction_prepare_data.py --work_dir 'compressLLM_baseline_wo-cl'

# todo: 不用compress loss看看下游任务效果呢
#python pre_prepare_data.py --work_dir 'compressLLM_long_text'
#python ./pre_trainer.py --work_dir 'compressLLM_long_text' --port 14529
#python ./trainer.py --work_dir 'compressLLM_test' --port 14529
#python ./pre_evaluator.py --work_dir 'compressLLM_baseline_wo-cl' --batch_size 1
#python ./instruction_trainer.py --work_dir 'compressLLM_test' --port 14525
#python ./instruction_evaluator.py --work_dir 'compressLLM_baseline_wo-cl' --batch_size 1

python ./trainer.py --work_dir 'compressLLM_test' --port 14523
python ./evaluator.py --work_dir 'compressLLM_test' --batch_size 1
#python ./instruction_trainer.py --work_dir 'compressLLM_test' --port 14525
#python ./instruction_trainer.py --work_dir 'compressLLM_test_pro' --port 14525
#python ./instruction_evaluator.py --work_dir 'compressLLM_test' --batch_size 1
#python ./instruction_evaluator.py --work_dir 'compressLLM_random_instruction_(pre-train-multi-lora)_multi-lora_lm' --batch_size 1




