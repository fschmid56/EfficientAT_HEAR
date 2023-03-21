import torch
from torch import nn
import torch.nn.functional as F
from torch import autocast
from contextlib import nullcontext


class MNHearWrapper(nn.Module):
    """
    Wraps a model into HEAR compatible format.
    Important functions that are called to extract embeddings:
        - get_scene_embeddings
        - get_timestamp_embeddings
    """
    def __init__(self, mel: nn.Module, net: nn.Module, mode=("clf1",), all_blocks=False, max_model_window=10000,
                 timestamp_window=160, timestamp_hop=50, scene_hop=2500,  scene_embedding_size=1487,
                 timestamp_embedding_size=1487):
        """
        @param mel: spectrogram extractor
        @param net: network module
        @param mode: determines which embeddings to use
        @param all_blocks: if true, uses all block output features as embeddings
        @param max_model_window: maximum clip length allowed by the model (milliseconds)
        @param timestamp_window: window size for timestamps (milliseconds)
        @param timestamp_hop: the hop length for timestamp embeddings (milliseconds)
        @param scene_hop: the hop length for scene embeddings (milliseconds)
        @param scene_embedding_size: required by HEAR API, size of embedding vector
        @param timestamp_embedding_size: required by HEAR API, size of embedding vector per timeframe
        """
        torch.nn.Module.__init__(self)
        self.mel = mel
        self.net = net
        self.device_proxy = nn.Parameter(torch.zeros(1))
        self.sample_rate = mel.sr  # required to be specified according to HEAR API
        self.timestamp_window = int(timestamp_window * self.sample_rate / 1000)  # in samples
        self.max_model_window = int(max_model_window * self.sample_rate / 1000)  # in samples
        self.timestamp_hop = int(timestamp_hop * self.sample_rate / 1000)  # in samples
        self.scene_hop = int(scene_hop * self.sample_rate / 1000)  # in samples
        self.scene_embedding_size = scene_embedding_size  # required by HEAR API
        self.timestamp_embedding_size = timestamp_embedding_size  # required by HEAR API
        self.mode = mode
        self.all_blocks = all_blocks
        self.nr_of_blocks = 15  # mobilenet has always 15 blocks

    def device(self):
        return self.device_proxy.device

    def forward(self, x):
        with torch.no_grad(), autocast(device_type=self.device().type) \
                if self.device().type.startswith("cuda") else nullcontext():
            specs = self.mel(x)
            mel_avgs = specs.detach().mean(dim=2)
            specs = specs.unsqueeze(1)
            x, features = self.net(specs)

        embed = torch.empty(0).to(x.device)

        if "mel_avgs" in self.mode:
            embed = torch.cat([embed, torch.flatten(mel_avgs, start_dim=1)], dim=1)
        if "clf1" in self.mode:
            embed = torch.cat([embed, features['classifier_features'][0]], dim=1)
        if "clf2" in self.mode:
            embed = torch.cat([embed, features['classifier_features'][1]], dim=1)
        if "logits" in self.mode:
            embed = torch.cat([embed, x.detach()], dim=1)
        if "se5" in self.mode:
            embed = torch.cat([embed, features['se_features'][0]], dim=1)
        if "se11" in self.mode:
            embed = torch.cat([embed, features['se_features'][1]], dim=1)
        if "se13" in self.mode:
            embed = torch.cat([embed, features['se_features'][2]], dim=1)
        if "se15" in self.mode:
            embed = torch.cat([embed, features['se_features'][3]], dim=1)

        if self.all_blocks:
            # index 0 would be in_conv
            for i in range(0, 16):
                if f"b{i}" in self.mode:
                    embed = torch.cat([embed, features['block_features'][i]], dim=1)
        else:
            # default block ids if not set otherwise by modifying argument 'collect_component_ids' in MobileNetV3.py
            if "b5" in self.mode:
                embed = torch.cat([embed, features['block_features'][0]], dim=1)
            if "b11" in self.mode:
                embed = torch.cat([embed, features['block_features'][1]], dim=1)
            if "b13" in self.mode:
                embed = torch.cat([embed, features['block_features'][2]], dim=1)
            if "b15" in self.mode:
                embed = torch.cat([embed, features['block_features'][3]], dim=1)
        return embed

    def get_scene_embeddings(self, audio):
        """
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed
                to the same length.
        Returns:
        embedding: A float32 Tensor with shape (n_sounds, model.scene_embedding_size).
        """
        n_sounds, n_samples = audio.shape
        # limit the maximal audio length to be fed at once to the model, long audio clips otherwise may exceed memory
        if n_samples <= self.max_model_window:
            embed = self.forward(audio.to(self.device()).contiguous())
            return embed
        # compute embeddings as average embeddings of 10 seconds snippets
        embeddings, timestamps = self.get_timestamp_embeddings(audio, window_size=self.max_model_window,
                                                               hop=self.scene_hop)
        return embeddings.mean(axis=1)

    def get_timestamp_embeddings(self, audio: torch.Tensor, window_size=None, hop=None, pad=None):
        """
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed
                to the same length.
        window_size, hop: only set when 'get_scene_embeddings' uses 'get_timestamp_embeddings' as fallback
        Returns:
        embedding: A float32 Tensor with shape (n_sounds, n_timestamps, model.timestamp_embedding_size).
        timestamps: A float32 Tensor with shape (`n_sounds, n_timestamps).
                    Centered timestamps in milliseconds corresponding to each embedding in the output.
        """
        if hop is None:
            hop = self.timestamp_hop
        if window_size is None:
            window_size = self.timestamp_window
        if pad is None:
            pad = window_size // 2
        audio = audio.cpu()
        n_sounds, n_samples = audio.shape
        audio = audio.unsqueeze(1)  # n_sounds, 1, (n_samples+pad*2)
        padded = F.pad(audio, (pad, pad), mode='reflect')
        padded = padded.unsqueeze(1)  # n_sounds, 1, (n_samples+pad*2)
        segments = F.unfold(padded, kernel_size=(1, window_size), stride=(1, hop)).transpose(-1, -2).transpose(0, 1)
        timestamps = []
        embeddings = []
        for i, segment in enumerate(segments):
            timestamps.append(i)
            embeddings.append(self.forward(segment.to(self.device())).cpu())

        timestamps = torch.as_tensor(timestamps) * hop * 1000. / self.sample_rate

        embeddings = torch.stack(embeddings).transpose(0, 1)  # now n_sounds, n_timestamps, timestamp_embedding_size
        timestamps = timestamps.unsqueeze(0).expand(n_sounds, -1)

        return embeddings, timestamps
