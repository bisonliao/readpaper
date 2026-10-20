### 目标：

1. 复现lift/can/square等等经典任务的训练
2. 使用robomimic这个现成的IL/RL库，完成自定义的环境/任务的训练

### 关键tips

#### 1、被验证过的超参数

下载数据后，运行 `generate_paper_configs.py` 会生成复现实验所需的配置文件。

```
# Assume datasets already exist in robomimic/../datasets folder. Configs will be generated under robomimic/exps/paper, and training results will be at /tmp/experiment_results when launching training runs.
$ python generate_paper_configs.py --output_dir /tmp/experiment_results

# Alternatively, specify where datasets exist, and specify where configs should be generated.
$ python generate_paper_configs.py --config_dir /tmp/configs --dataset_dir /tmp/datasets --output_dir /tmp/experiment_results
```

详细见文档：

```
https://robomimic.github.io/docs/datasets/robomimic_v0.1.html#reproducing-results
```

这个太重要了。执行后生成了一堆配置文件，分门别类非常清晰。