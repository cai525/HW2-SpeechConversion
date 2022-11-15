# Voice-Conversion
完成了《人类语言处理》的HW2.1。参考代码见 [HW提供的例程](https://github.com/BogiHsu/Voice-Conversion) 和 [jjery2243542/voice_conversion (github.com)](https://github.com/jjery2243542/voice_conversion)

## 依赖
- h5py

-  scipy>=1.7.3

- numpy

- torch>=1.10.0+cu102

- librosa>=0.9.2

- matplotlib>=3.5.1

- ffmpeg-python

- tensorboardx>=2.5.1

- tqdm


## 预处理

需要采用预处理文件夹中的两个文件：

- preprocess/make_dataset_vctk.py : 将数据划分为训练集和测试集，并将数据保存为.h5格式文件

- preprocess/make_single_samples.py : 获取对数据集中数据的采样，并以json文件格式保存


注：关于该项目json文件的格式

eg:

```
    {
        "speaker": 0,
        "i": "1/131",
        "t": 84
    }
```

- “speaker”代表说话者
- "i"代表从h5文件中读取的语言文件，"1/131"代表P1的131号音频
- "t" 指定开始采样的位置，从该位置后采样 **seg_len** 个帧



## 训练

使用train.py训练。参考[jjery2243542/voice_conversion (github.com)](https://github.com/jjery2243542/voice_conversion) 中的输入参数。

> 说明：
>
> 1. 本项目额外添加了命令行参数 '-last_iter',即中断训练前上一次的训练轮数，用于在中断训练后恢复到之前的位置继续训练。
> 2. 中断训练后重新加载时，记得在train.py 中把已经训练结束训练过程注释掉



## 使用

训练完成后，使用 example.ipynb 可以快速查看单个音频的转换效果。批量转换需使用convert.py。

> 注：运行 example.ipynb 所需的全部文件已上传，该文件也可在训练前直接运行。

