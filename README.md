# Low-complexity Audio Embedding Extractors

This repository aims to provide **low-complexity general-purpose audio embedding extractors (GPAEE)**.
The corresponding paper [Low-complexity Audio Embedding Extractors](https://arxiv.org/pdf/2303.01879.pdf) is submitted to
[Eusipco 2023](https://eusipco2023.org/). The models used as low-complexity GPAEE are pre-trained on AudioSet using Knowledge
Distillation from Transformers. The pre-training is described in detail in the paper 
[Efficient Large-Scale Audio Tagging Via Transformer-To-CNN Knowledge Distillation](https://arxiv.org/pdf/2211.04772.pdf) 
accepted to [ICASSP 2023](https://2023.ieeeicassp.org/). The pre-trained models are available in the repository [EfficientAT](https://github.com/fschmid56/EfficientAT).

GPAEEs have to process the high-dimensional raw audio signals only once,
while shallow downstream classifiers can solve different audio tasks based on the extracted low-dimensional embeddings. 
This procedure can save lots of compute when multiple tasks need to be solved in parallel based on an audio input stream.

However, extracting high-quality audio embeddings is usually accomplished using complex transformers or large CNNs, 
for example, PaSST [1,2] and PANNs [3]. In this work, we investigate two research questions:

* How can well-performing general-purpose audio representations be obtained from a CNN?
* How is the model complexity related to the quality of extracted embeddings?

We show that low-complexity CNNs can extract high-quality audio embeddings, paving the way for applying GPAEEs
on edge devices. As shown in the Figure below, a concrete application would be a low-complexity GPAEE running on a mobile phone
and producing audio embeddings for the continuous audio stream received by the mobile phone. Based on the embeddings, 
comparably cheap MLP classifiers can solve a variety of different tasks, e.g. identifying the speaker, recognizing the music genre 
and identifying the acoustic scene.

![Application](/images/mobile.png)

## Main Results

We evaluate the quality of extracted embeddings for all models on the [HEAR benchmark](https://hearbenchmark.com/), 
described in detail in [this paper](https://arxiv.org/pdf/2203.03022.pdf) [4]. In short, HEAR comprises 19 tasks with short and
long time spans, covering different audio domains such as speech, music and environmental sounds. The range of tasks is 
extremely broad, ranging from detecting the location of a gunshot to discriminating normal vs. queen-less beehives to 
classifying emotion in speech. In the first step, the GPAEE to be evaluated must generate the embeddings for all sound clips
in all tasks. In the second step, a shallow MLP is trained for each task based on the generated embeddings. The performance
of the MLPs on the downstream tasks corresponds to the quality of extracted embeddings.

We normalize each task score by the max score achieved by a model in the official HEAR challenge. To express the benchmark
result as a single number, we average the normalized scores across all tasks.

In the paper (table below), we show that a combination of mid- and low-level features work best as general-purpose audio
embeddings extracted from a CNN. Mid-level features are extracted from intermediate convolutional layers by using global channel
pooling. Low-level features are generated by computing the average value of a Mel band across all time frames. Concatenating
low- and mid-level features combines low-level pitch information with an abstract feature representation.

![Model Comparison](/images/features.png)

We scale our pre-trained models by network width (number of channels) to receive GPAEE of varying complexity. The plot below
shows that our proposed models have an excellent performance-complexity tade-off compared to well-performing challenge submissions.
For instance, our smallest model with 120k parameters extracts embeddings of quality comparable to PaSST [1,2] and 
outperforming PANNs [3] (both around 80M parameters).

![Model Comparison](/images/model_comp.png)

We also categorize the HEAR tasks into **Speech**, **Music** and **General** sounds and compare the distribution of normalized scores for each category between
the models (see figure below). The plot below shows that overall mn30 and mn10 perform favourably against all other single models
submitted to the challenge, while the tiny mn01 is still very competitive. Our models push the max challenge score to a new
level on multiple tasks including *Beijing Opera Percussion*, *Mridingham Tonic*, *Mridingham Stroke*, *GTZAN Genre*,
*Vocal Imitations* and *ESC-50*.

![Category Analysis](/images/category_analysis.png)


# Setup
### Installation

The setup has been tested using python 3.8. Create and activate a conda environment:

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
from hear_mn.old_configs import mn40_ext_e1_l

seconds = 20
audio = torch.ones((1, 32000 * seconds)) * 0.5
wrapper = mn40_ext_e1_l.load_model().cuda()

embed, time_stamps = wrapper.get_timestamp_embeddings(audio)
print(embed.shape)
embed = wrapper.get_scene_embeddings(audio)
print(embed.shape)
```

## References

[1] Khaled Koutini, Jan Schlüter, Hamid Eghbal-zadeh, and Gerhard Widmer, “Efficient Training of Audio Transformers with Patchout,” in Interspeech, 2022.

[2] Koutini, K., Masoudian, S., Schmid, F., Eghbal-zadeh, H., Schlüter, J., & Widmer, G. (2022). Learning General Audio Representations with Large-Scale Training of Patchout Audio Transformers. arXiv preprint arXiv:2211.13956.

[3] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley, “Panns: Large-scale pretrained audio neural networks for audio pattern recognition,” IEEE ACM Trans. Audio Speech Lang. Process., 2020.

[4] J. Turian, J. Shier, H. R. Khan, B. Raj, B. W. Schuller, C. J. Steinmetz,
C. Malloy, G. Tzanetakis, G. Velarde, K. McNally, M. Henry, N. Pinto,
C. Noufi, C. Clough, D. Herremans, E. Fonseca, J. H. Engel, J. Salamon,
P. Esling, P. Manocha, S. Watanabe, Z. Jin, and Y. Bisk, “HEAR: holistic
evaluation of audio representations,” in NeurIPS 2021 Competitions and
Demonstrations Track. PMLR, 2021.