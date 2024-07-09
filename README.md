# mlllm
 Enhance LLM memory with self-supervised objectives for in-memory learning.


## 环境配置
```
git clone https://github.com/1azybug/mlllm.git
cd mlllm
conda create -n forget python=3.10 -y
conda activate forget
conda install pytorch==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 训练

```
cd compress
bash train.sh
```

## 超参数更改
修改 ./compress/CompressLLM/config.json
![config](./config.png "config")

## Tips
处理数据的过程会比较慢，可以一边训练一边处理下一次训练用到的数据：
```
cd compress
python prepare_data.py --work_dir compressLLM_len-500_ratio_5
```

处理完后
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./trainer.py --work_dir compressLLM_len-500_ratio_5 --port 12314
```