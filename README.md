# 情绪式对话生成系统

这是一个能够在文本对话中表达情感的对话系统。[在线尝试！](https://cakechat.replika.ai/)

![Demo](https://user-images.githubusercontent.com/764902/34832660-92570bfe-f6fe-11e7-9802-db2f8730a997.png)

它用[Theano](http://deeplearning.net/software/theano/) 和 [Lasagne](https://github.com/Lasagne/Lasagne)编写，使用5种不同情绪的端到端训练嵌入来产生由给定情绪限制的反应。

代码是灵活的，并允许通过为训练数据中的一些样本定义的任意分类变量来调节响应。例如，您可以训练基于角色的个人神经对话模型或创建一个没有外部记忆的情感聊天机器人。


## 目录

  * [网络架构和特征](#network-architecture-and-features)
  * [快速开始](#quick-start)
  * [设置](#setup)
    * [Docker](#docker)
      * [CPU设置](#cpu-only-setup)
      * [GPU设置](#gpu-enabled-setup)
    * [手动设置](#manual-setup)
  * [获取模型](#getting-the-model)
    * [使用预训练模型](#using-a-pre-trained-model)
    * [训练你自己的模型](#training-your-own-model)
    * [现有的训练数据集](#existing-training-datasets)
  * [运行系统](#running-the-system)
    * [Local-HTTP-server](#local-http-server)
      * [HTTP-server-API-description](#http-server-api-description)
    * [Gunicorn HTTP\-server](#gunicorn-http-server)
    * [电报机器人](#telegram-bot)
  * [存储库概述](#repository-overview)
    * [重要工具](#important-tools)
    * [重要认证设置](#important-configuration-settings)
  * [用例示例](#example-use-cases)
  * [参考](#references)
  * [学分和支持](#credits--support)
  * [执照](#license)


## 网络架构和功能

![网络架构](https://user-images.githubusercontent.com/4047271/34774960-71c7bfa0-f622-11e7-812b-cdb84472577a.png)


* 模型：
用于处理深层对话上下文的分层递归编码器 - 解码器（HRED）架构[7]
具有GRU细胞的多层RNN。话语级编码器的第一层始终是双向的。
思想向量在每个解码步骤被馈入解码器。
解码器可以在任何字符串标签上进行调节。例如：情感标签或一个人说话的身份。
Word嵌入层：
可以使用在您自己的语料库中训练的w2v模型进行初始化。
嵌入层可以保持固定，与网络的所有其他权重一起进行微调。
解码
4种不同的响应生成算法：“采样”，“波束搜索”，“采样重新排序”和“波束搜索重新排序”。根据对数似然或MMI标准[3]执行对生成的候选者的重新排名。有关详情，请参阅配置设置说明
指标：
困惑
n-gram不同的指标调整到样本大小[3]。
模型样本与某些固定数据集之间的词汇相似性。词汇相似性是模型生成的响应的TF-IDF向量与数据集中的令牌之间的余弦距离。
排名指标：平均精确度和平均召回量@ k。[8]
快速开始
运行仅CPU的预构建docker镜像并启动在8080端口上为模型提供服务的CakeChat:

```(bash)
docker run --name cakechat-dev -p 127.0.0.1:8080:8080 -it lukalabs/cakechat:latest \
    bash -c "python bin/cakechat_server.py"
```

(Or) using the GPU-enabled image:

```(bash)
nvidia-docker run --name cakechat-gpu-dev -p 127.0.0.1:8080:8080 -it lukalabs/cakechat-gpu:latest \
    bash -c "USE_GPU=0 python bin/cakechat_server.py"
```

你可以尝试运行 `python tools/test_api.py -f localhost -p 8080 -c "Hi! How are you?"` 于命令行.

## 设置

### Docker

这是设置环境和安装所有依赖项的最简单方法。

#### CPU设置

1. 安装 [Docker](https://docs.docker.com/engine/installation/)

2. 安装docker镜像

构建cpu镜像:

```(bash)
docker build -t cakechat:latest -f dockerfiles/Dockerfile.cpu dockerfiles/
```

3. 打开容器

在仅CPU环境中运行docker容器
```(bash)
docker run --name <CONTAINER_NAME> -it cakechat:latest
```

#### GPU设置

1. 安装[nvidia-docker](https://github.com/NVIDIA/nvidia-docker) .

2. 构建支持GPU的docker镜像:

```(bash)
nvidia-docker build -t cakechat-gpu:latest -f dockerfiles/Dockerfile.gpu dockerfiles/
```

3. 打开容器

在启用GPU的环境中运行docker容器:

```(bash)
nvidia-docker run --name <CONTAINER_NAME> -it cakechat-gpu:latest
```

你可以训练你自己的模型并开始聊天.


### 自定义

如果您不想处理docker图像和容器，您可以随时运行(用 `sudo`, `--user` 或者你内部 **[virtualenv](https://virtualenv.pypa.io/en/stable/)**):

```(bash)
pip install -r requirements.txt
```

很可能这将完成这项工作.
**NB:** 此方法仅提供仅CPU环境。 要获得GPU支持，您需要构建和安装 **[libgpuarray](http://deeplearning.net/software/libgpuarray/installation.html)** 通过你自己 (看 [Dockerfile.gpu](dockerfiles/Dockerfile.gpu) for example).


## 获取模型

### 使用预训练模型

运行 `python tools/download_model.py` 来下载我们的预训练模型.

这个模型是使用 **context size 3** 
编码序列包含 **30 tokens or less** 和解码序列 **32 tokens or less**.
编码和解码序列都包含 **2 GRU layers** 和 **512 hidden units** .
该模型在Twitter预处理的会话数据上进行了训练。
为了清理数据，我们删除了网址，转发和引文。
我们还删除了没有正常单词或标点符号的提及和主题标签
并筛选出包含30个以上令牌的所有邮件。
然后我们用我们的情绪分类器标出每个话语，预测其中一个
5种情绪：“中性”，“快乐”，“愤怒”，“悲伤”和“恐惧”。
用你可以使用的情感来标记自己的语料库，例如[DeepMoji工具]（https://github.com/bfelbo/DeepMoji）
或者你拥有的任何其他情绪分类器。

#### 从文件初始化模型权重
 对于一些工具 (例如 [`tools/train.py`](tools/train.py)) 您可以通过参数指定模型初始化权重的路径 `--init_weights` .
权重可以来自经过训练的模型或来自具有不同体系结构的模型。
在后一种情况下，可以保留模型的一些参数而不进行初始化：
如果参数的名称和形状是，则将使用保存的值初始化参数
与保存的参数相同，否则参数将保持其默认初始化权重。
从 `load_weights` 中获得功能的细节.

### 训练自己的模型

1. 将您的培训文本语料库添加到 [`data/corpora_processed/`](data/corpora_processed/).
语料库文件的每一行都应该是一个JSON对象，其中包含按时间顺序排序的对话消息列表。
代码完全与语言无关 - 您可以在数据集中使用任何unicode文本。
请参阅我们的虚拟语料库以查看输入格式 [`data/corpora_processed/train_processed_dialogs.txt`](data/corpora_processed/train_processed_dialogs.txt).

 2.以下数据集用于验证和提前停止:

* [`data/corpora_processed/val_processed_dialogs.txt`](data/corpora_processed/val_processed_dialogs.txt)(假的例子) - 对于上下文敏感数据集
* [`data/quality/context_free_validation_set.txt`](data/quality/context_free_validation_set.txt) - 用于无上下文的验证数据集
* [`data/quality/context_free_questions.txt`](data/quality/context_free_questions.txt) - 用于生成记录和计算不同度量标准的响应
* [`data/quality/context_free_test_set.txt`](data/quality/context_free_test_set.txt) - 用于计算训练模型的度量，例如， 排名指标
3. 设置训练参数 [`cakechat/config.py`](cakechat/config.py).
看 [configuration settings description](#important-configuration-settings) 更多细节.
4. 运行 `python tools/prepare_index_files.py` 使用训练语料库中的标记和条件构建索引文件.
5. 运行 `python tools/train.py`. 别忘记设置 `USE_GPU=<GPU_ID>` 环境变量 (用 GPU_ID 到 **nvidia-smi**) 如果你使用GPU.
使用`SLICE_TRAINSET=N` 在训练数据的前N个样本的子集上训练模型，以加快预处理以进行调试.
6. 您还可以设置`IS_DEV = 1`以启用“开发模式”。 它使用减少数量的模型参数（减少隐藏层维度，令牌序列的输入和输出大小等），执行详细日志记录并禁用Theano图优化。 使用此模式进行调试。
7.模型的权重将保存在`data / nn_models /`中.
### 现有的培训数据集
您可以在任何可用的文本会话数据集上训练对话模型。 可以在此处找到现有会话数据集的精彩概述: https://breakend.github.io/DialogDatasets/


## 运行系统

### 本地HTTP服务器

运行使用给定输入消息（上下文）处理HTTP请求的服务器，并返回模型的响应消息:

```(bash)
python bin/cakechat_server.py
```
如果要使用某个GPU，请指定`USE_GPU=<GPU_ID>`环境变量。

等到模型编译完毕.

**别忘记运行 [`tools/download_model.py`](tools/download_model.py) 在运行之前 [`bin/cakechat_server.py`](bin/cakechat_server.py) 如果你想用我们预先训练过的模型启动API。**

为确保一切正常，请在以下对话中测试模型:

> – Hi, Eddie, what's up?  
> – Not much, what about you?  
> – Fine, thanks. Are you going to the movies tomorrow?


```(bash)
python tools/test_api.py -f 127.0.0.1 -p 8080 \
    -c "Hi, Eddie, what's up?" \
    -c "Not much, what about you?" \
    -c "Fine, thanks. Are you going to the movies tomorrow?"
```

#### HTTP服务器API说明

##### /cakechat_api/v1/actions/get_response
JSON parameters are:

|Parameter|Type|Description|
|---|---|---|
|context|list of strings|List of previous messages from the dialogue history (max. 3 is used)|
|emotion|string, one of enum|One of {'neutral', 'anger', 'joy', 'fear', 'sadness'}. An emotion to condition the response on. Optional param, if not specified, 'neutral' is used|

##### 响应
```
POST /cakechat_api/v1/actions/get_response
data: {
 'context': ['Hello', 'Hi!', 'How are you?'],
 'emotion': 'joy'
}
```

##### 响应完成
```
200 OK
{
 'response': 'I\'m fine!'
}
```

### Gunicorn HTTP-server

We recommend to use [Gunicorn](http://gunicorn.org/) for serving the API of your model at a production scale.

Run a server that processes HTTP-queries with input messages and returns response messages of the model:

```(bash)
cd bin && gunicorn cakechat_server:app -w 1 -b 127.0.0.1:8080 --timeout 2000
```

You may need to install gunicorn from pip: `pip install gunicorn`.


### 电报机器人

You can also test your model in a Telegram bot:
[create a telegram bot](https://core.telegram.org/bots#3-how-do-i-create-a-bot) and run

`python tools/telegram_bot.py --token <YOUR_BOT_TOKEN>`


## 存储库概述
* `cakechat/dialog_model/` - 包含计算图，训练程序和其他模型实用程序
* `cakechat/dialog_model/inference/` - 响应生成算法
* `cakechat/dialog_model/quality/` - 度量计算和日志记录的代码
* `cakechat/utils/` - 用于文本处理，w2v培训等的实用程序
* `cakechat/api/` - 运行http服务器的函数：API配置，错误处理
* `tools/` - 用于培训，测试和评估模型的脚本


### 重要工具

* [`bin/cakechat_server.py`](bin/cakechat_server.py) - 
运行HTTP服务器，该服务器返回给定对话框上下文和情绪的模型的响应消息。 看到 [run section](#gunicorn-http-server) 更多资料.
* [`tools/train.py`](tools/train.py) - 
在您的数据上训练模型。 您可以通过指定模型参数初始化权重的路径 `--init_weights` . 也可以使用标记用于训练"\*-reranking" 响应生成算法中使用的模型，以获得更准确的预测。
* [`tools/prepare_index_files.py`](tools/prepare_index_files.py) - 
准备最常用的令牌和条件的索引。 在训练模型之前使用此脚本。
* [`tools/quality/ranking_quality.py`](tools/quality/ranking_quality.py) - 
计算对话模型的排名度量。
* [`tools/quality/prediction_distinctness.py`](tools/quality/prediction_distinctness.py) - 
计算不同-对话模型的指标。
看 [features section](#network=architecture=and-features) 更多关于指标资料.
* [`tools/quality/condition_quality.py`](tools/quality/condition_quality.py) - 
根据条件值计算数据的不同子集的度量.
* [`tools/generate_predictions.py`](tools/generate_predictions.py) - 
评估模型。 在给定的对话框上下文集上生成对话模型的预测，然后计算度量。
请注意，您应该具有反向模型 `data/nn_models` 目录，如果你想使用 "\*-reranking" 预测模式。
* [`tools/generate_predictions_for_condition.py`](tools/generate_predictions_for_condition.py) - 
生成给定条件值的预测。
* [`tools/test_api.py`](tools/test_api.py) - 
用于将请求发送到正在运行的HTTP服务器的示例代码。
* [`tools/download_model.py`](tools/download_model.py) - 
下载预先训练的模型和与之关联的索引文件。 还要编译整个模型一次以创建Theano缓存。
* [`tools/telegram_bot.py`](tools/telegram_bot.py) - 
运行使用训练模型的电报机器人。

### 重要的配置设置

网络体系结构，培训，预测和记录步骤的所有配置参数都在中定义[`cakechat/config.py`](cakechat/config.py).
HTTP服务器中使用的一些推理参数在中定义 [`cakechat/api/config.py`](cakechat/api/config.py).

* 网络架构和规模
    * `HIDDEN_LAYER_DIMENSION` 是定义循环图层中隐藏单元数的主要参数.
    * `WORD_EMBEDDING_DIMENSION` 和 `CONDITION_EMBEDDING_DIMENSION` 定义每个隐藏单元的数量
     令牌/条件映射到。
     它们一起汇总到传递给编码器RNN的输入矢量的维度。
    * 解码器输出层的单元数由字典中的令牌数定义
     tokens_index目录。
* 解码算法:
    * `PREDICTION_MODE_FOR_TESTS` 定义如何生成模型的响应。 选项如下:
        -  **sampling** – response is sampled from output distribution token-by-token. 
        对于每个标记，在采样之前执行温度变换。
         您可以通过调整来控制温度值 `DEFAULT_TEMPERATURE` 参数.
        - **sampling-reranking** – multiple candidate-responses are generated using sampling procedure described above.
        之后候选人根据他们的MMI得分排名[<sup>\[3\]</sup>](#f3)
        您可以通过选择来调整参数此模式 `SAMPLES_NUM_FOR_RERANKING` 和 `MMI_REVERSE_MODEL_SCORE_WEIGHT` .
        - **beamsearch** – candidates are sampled using [beam search algorithm](https://en.wikipedia.org/wiki/Beam_search).
       根据由波束搜索过程计算的对数似然分数对候选者进行排序。
        - **beamsearch-reranking** – same as above, but the candidates are re-ordered after the generation in the same way as
        在采样 - 重新排名模式下。
        
    请注意，还有其他参数会影响响应生成过程。
    See `REPETITION_PENALIZE_COEFFICIENT`, `NON_PENALIZABLE_TOKENS`, `MAX_PREDICTIONS_LENGTH`.

## 用例示例

通过在数据集条目中提供其他条件标签，您可以构建以下模型：
* [A Persona-Based Neural Conversation Model][5] — 允许对角色ID进行条件响应以使其在词汇上与给定角色的语言风格相似的模型。
* [Emotional Chatting Machine][4]-like model — 允许调节情绪反应以提供情绪风格（愤怒，悲伤，快乐等）的模型。
* [Topic Aware Neural Response Generation][6]-like model  - 一种模型，允许对特定主题的响应进行条件化以保持主题感知对话。
要使用这些额外条件，请参阅本节 [Training your own model](#training-your-own-model). 只需设置“条件”字段即可[training set](data/corpora_processed/train_processed_dialogs.txt) 以下之一: **persona ID**, **emotion** or **topic**标签，更新索引文件并开始培训。
## 参考

* <a name="f1"/><sup>\[1\]</sup> [神经对话模型][1]
* <a name="f2"/><sup>\[2\]</sup> [如何不评估你的对话系统][2]
* <a name="f3"/><sup>\[3\]</sup> [神经对话模型的多样性促进目标函数][3]
* <a name="f4"/><sup>\[4\]</sup> [情绪聊天机：内部和外部记忆的情感对话][4]
* <a name="f5"/><sup>\[5\]</sup> [基于角色的神经对话模型][5]
* <a name="f6"/><sup>\[6\]</sup> [主题意识神经响应生成][6]
* <a name="f7"/><sup>\[7\]</sup> [用于生成上下文感知查询建议的分层递归编码器 - 解码器][7]
* <a name="f8"/><sup>\[8\]</sup> [口语对话系统用户仿真技术的定量评估][8]

[1]: https://arxiv.org/pdf/1506.05869.pdf
[2]: https://arxiv.org/pdf/1603.08023.pdf
[3]: https://arxiv.org/pdf/1510.03055.pdf
[4]: https://arxiv.org/pdf/1704.01074.pdf
[5]: https://arxiv.org/pdf/1603.06155.pdf
[6]: https://arxiv.org/pdf/1606.08340v2.pdf
[7]: https://arxiv.org/pdf/1507.02221.pdf
[8]: http://mi.eng.cam.ac.uk/~sjy/papers/scgy05.pdf

## 学分和支持
它由 [Replika](https://replika.ai)开发和维护: [Michael Khalman](https://github.com/mihaha), [Nikita Smetanin](https://github.com/nsmetanin), [Artem Sobolev](https://github.com/artsobolev), [Nicolas Ivanov](https://github.com/nicolas-ivanov), [Artem Rodichev](https://github.com/rodart) and [Denis Fedorenko](https://github.com/sadreamer). Demo by [Oleg Akbarov](https://github.com/olegakbarov), [Alexander Kuznetsov](https://github.com/alexkuz) and [Vladimir Chernosvitov](http://chernosvitov.com/).

可以在此处跟踪所有问题和功能请求 - [GitHub Issues](https://github.com/lukalabs/cakechat/issues).
