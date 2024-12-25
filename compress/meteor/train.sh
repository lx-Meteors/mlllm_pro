
# todo: 不用compress loss看看下游任务效果呢
#python pre_prepare_data.py --work_dir 'compressLLM_long_text'
#python ./pre_trainer.py --work_dir 'compressLLM_long_text' --port 14529
#python ./trainer.py --work_dir 'compressLLM_long_text' --port 14529
#python ./pre_evaluator.py --work_dir 'compressLLM_long_text' --batch_size 1
#python ./instruction_trainer.py --work_dir 'compress_test_v2' --port 14525
#python ./instruction_evaluator.py --work_dir 'compressLLM_baseline_wo-cl' --batch_size 1


python prepare_data.py --work_dir 'forzen-lora'
python ./trainer.py --work_dir 'forzen-lora' --port 14529
python ./pre_evaluator.py --work_dir 'forzen-lora' --batch_size 1
# python instruction_prepare_data.py --work_dir 'forzen-lora'

python ./trainer.py --work_dir 'semantic' --port 14528
python ./pre_evaluator.py --work_dir 'forzen-lora' --batch_size 1
python ./instruction_trainer.py --work_dir 'forzen-lora'  --port 14527
python ./instruction_evaluator.py --work_dir 'forzen-lora' --batch_size 1