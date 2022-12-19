# EfficientAT@HEAR21Benchmark
This repository aims at evaluating the extracted audio representations of the efficient audio tagging models presented 
in the paper [Efficient Large-Scale Audio Tagging Via Transformer-To-CNN Knowledge Distillation](https://arxiv.org/pdf/2211.04772.pdf).
Examples, code and details can be found in the repository [EfficientAT](https://github.com/fschmid56/EfficientAT).

We evaluate models on the [Holistic Evaluation of Audio Representations Benchmark](https://hearbenchmark.com/). The benchmark
proceeds in two steps. Firstly, audio embeddings are generated for all 19 tasks. Secondly, a shallow MLP classifier
is trained on the tasks using the generated embeddings as input.

Checkout the [HEAR paper](https://arxiv.org/abs/2203.03022) for all details of the benchmark.

### Results in short

![Model Comparison](/images/s4_1.png)

Comparison of our pre-trained CNN model (`mn40_as_ext_e1_l_all_se`) to PASST [1] and PANNs [2] across three 
task categories. 

# Setup
### Installation

Setup has been tested using python 3.8. Create and activate a conda environment:

```
conda create --name hear python=3.8
conda activate hear
```

Install HEAR validator:

```
pip install hearvalidator
```

Install HEAR evaluation kit:

```
pip3 install heareval
```

Install the package contained in this repository: 

```
pip install -e 'git+https://github.com/fschmid56/EfficientAT_HEAR@0.0.1#egg=hear_mn' 
```

Install the exact torch, torchvision and torchaudio versions we tested our setup with (on a CUDA 11.1 system):

```
pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
```

### Task setup

To download the data and setup the tasks follow the [official HEAR guideline](https://hearbenchmark.com/hear-tasks.html).

### Validate setup and model

To check whether a model is correctly wrapped in the interface for the HEAR challenge, run the following to test e.g. the 
module `mn40_as_ext_e1_l`:

```
hear-validator hear_mn.mn40_ext_e1_l
```

### Generate embeddings for all tasks

```
python3 -m heareval.embeddings.runner hear_mn.mn40_ext_e1_l  --tasks-dir <path to tasks>
```

###  Run evaluation procedure

To train the shallow MLP classifier on the embeddings, run the following:

```
python3 -m heareval.predictions.runner embeddings/hear_mn.mn40_ext_e1_l/*
```

## Obtain Embeddings

All models follow the HEAR interface given the following 3 methods:

* load_model() 
* get_scene_embeddings(audio)
* get_timestamp_embeddings(audio)

These can for instance be used as follows:

```python
import torch
from hear_mn import mn40_ext_e1_l

seconds = 20
audio = torch.ones((1, 32000 * seconds))*0.5
wrapper = mn40_ext_e1_l.load_model().cuda()

embed, time_stamps = wrapper.get_timestamp_embeddings(audio)
print(embed.shape)
embed = wrapper.get_scene_embeddings(audio)
print(embed.shape)
```

# Results

We experiment and compare with the following variation of embeddings:
* Models of varying complexity
* `e1`: using features after adaptive average pooling (after conv, before linear layers)
* `clf`: using features after classifier linear layer
* `l`: adding logits to embeddings (527 AudioSet classes)
* `se`: adding bottleneck representation of subset of Squeeze-and-Excitation layers to embeddings
* `all_se`: adding bottleneck representation of all Squeeze-and-Excitation layers to embeddings
* `mels`: adding average of mel bands to embeddings
* `block`: perform adaptive average pooling over subset of feature maps and add to embeddings

We split the tasks into the three categories `speech`, `music` and `general` sounds as done in [3]. We compare our models to 
PASST [1] (state-of-the-art transformer) as well as PANNs [2] (very popular pre-trained CNNs). 

Our default setting is `e1_l` meaning that we concatenate the features obtained after adaptive average pooling with the 
logits.

We normalize all scores by the maximal test score a model achieved in the official challenge.

### Model complexity

![Model Complexity](/images/s0.png)

* The small model `mn04_as_e1_l` (0.98 Mil. params) performs exceptionally well against larger models (PaSST: 86 Mil. params,
PANNs: 80 Mil. params, mn40_as_ext_e1_l: 68 Mil. params, mn10_as_e1_l: 4.88 Mil. params)
* `mn40_as_ext_e1_l` outperforms PANNs and is slightly worse compared to PaSST

### Logits and Classifier Embeddings (`l` and `clf`)

![Model Complexity](/images/s1.png)

* Using `clf` is slightly better than `e1`
* Using no logits is slightly better than `l`

### Adding Squeeze-and-Excitation bottleneck representation (`se`) and pooled feature maps (`block`)

![Model Complexity](/images/s2.png)

* Adding `se` and `block` clearly improve performance
* Largest improvement in speech (lower level features helpful)
* `all_se` is slightly better than `block`

### Adding `mels`

![Model Complexity](/images/s3.png)

* Adding average mel band value does not make a lot of difference (except for task *MAESTRO 5h*)

### Comparing `mn40_as_ext_e1_l_all_se`, `PANNs` and `PASST` on an individual task basis:

![Model Complexity](/images/s4_2.png)

* `mn40_as_ext_e1_l_all_se` outperforms PaSST on 17/19 tasks
* `mn40_as_ext_e1_l_all_se` outperforms PANNs on 15/16 tasks
* Embedding Dimension for `mn40_as_ext_e1_l_all_se` (8423) is much higher in this settings than for 
PaSST (1295) and PANNs (2048).
* `mn40_as_ext_e1_l_all_se` achieves new best challenge results on the tasks *ESC-50* and *GTZAN Genre*

## References

[1] Khaled Koutini, Jan Schlüter, Hamid Eghbal-zadeh, and Gerhard Widmer, “Efficient Training of Audio Transformers with Patchout,” in Interspeech, 2022.

[2] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley, “Panns: Large-scale pretrained audio neural networks for audio pattern recognition,” IEEE ACM Trans. Audio Speech Lang. Process., 2020.

[3] Koutini, K., Masoudian, S., Schmid, F., Eghbal-zadeh, H., Schlüter, J., & Widmer, G. (2022). Learning General Audio Representations with Large-Scale Training of Patchout Audio Transformers. arXiv preprint arXiv:2211.13956.