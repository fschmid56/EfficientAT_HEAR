import torch
from torch import nn
import torch.nn.functional as F
from torch import autocast
from contextlib import nullcontext


class MNHearWrapper(nn.Module):
    def __init__(self, mel: nn.Module, net: nn.Module, max_model_window=10000, timestamp_window=160, timestamp_hop=50,
                 scene_hop=2500, scene_embedding_size=1487, timestamp_embedding_size=1487,
                 mode="all", add_se_features=False, add_block_features=False, add_classifier_features=False,
                 add_averaged_mels=False):
        """
        @param mel: spectrogram extractor
        @param net: network module
        @param max_model_window: maximum clip length allowed by the model (milliseconds).
        @param timestamp_hop: the hop length for timestamp embeddings (milliseconds).
        @param scene_hop: the hop length for scene embeddings (milliseconds).
        @param scene_embedding_size: required by HEAR API
        @param timestamp_embedding_size: required by HEAR API
        @param mode: "all", "embed_only", "logits"
        @param embed_mode: "last", "second_last", "all"
        """
        torch.nn.Module.__init__(self)
        self.mel = mel
        self.net = net
        self.device_proxy = nn.Parameter(torch.zeros((1)))
        self.sample_rate = mel.sr  # required to be specified according to HEAR API
        self.timestamp_window = int(timestamp_window * self.sample_rate / 1000)  # in samples
        self.max_model_window = int(max_model_window * self.sample_rate / 1000)  # in samples
        self.timestamp_hop = int(timestamp_hop * self.sample_rate / 1000)  # in samples
        self.scene_hop = int(scene_hop * self.sample_rate / 1000)  # in samples
        self.scene_embedding_size = scene_embedding_size  # required by HEAR API
        self.timestamp_embedding_size = timestamp_embedding_size  # required by HEAR API
        self.mode = mode
        self.add_se_features = add_se_features
        self.add_block_features = add_block_features
        self.add_classifier_features = add_classifier_features
        self.add_averaged_mels = add_averaged_mels

    def device(self):
        return self.device_proxy.device

    def forward(self, x):
        with torch.no_grad(), autocast(device_type=self.device().type) \
                if self.device().type.startswith("cuda") else nullcontext():
            specs = self.mel(x)
            specs = specs.unsqueeze(1)
            x, features = self.net(specs)
            features, additional_features = features
        x, features = x.float(), features.float()
        if self.mode == "all":
            embed = torch.cat([x, features], dim=1)
        elif self.mode == "embed_only":
            embed = features
        elif self.mode == "logits":
            embed = x
        elif self.mode == "none":
            embed = torch.empty(0).to(x.device)
        else:
            raise RuntimeError(f"mode='{self.mode}' is not recognized not in: all, embed_only, logits")

        if self.add_averaged_mels:
            embed = torch.cat([embed, torch.mean(specs, dim=(1, 3))], dim=1)

        if self.add_classifier_features:
            embed = torch.cat([embed, additional_features['classifier_features'][0]], dim=1)

        if self.add_block_features:
            block_features = additional_features['block_features']
            # first baseline - take channel-wise mean
            for block_feature in block_features:
                embed = torch.cat([embed, F.adaptive_avg_pool2d(block_feature, (1, 1)).squeeze(2).squeeze(2)], dim=1)

        if self.add_se_features:
            se_features = additional_features['se_features']
            # first baseline - take channel-wise mean
            for se_feature in se_features:
                embed = torch.cat([embed, se_feature], dim=1)

        return embed

    def get_scene_embeddings(self, audio):
        """
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
        model: Loaded Model.
        Returns:
        embedding: A float32 Tensor with shape (n_sounds, model.scene_embedding_size).
        """
        n_sounds, n_samples = audio.shape
        # limit the maximal audio length to be fed at once to the model
        if n_samples <= self.max_model_window:
            embed = self.forward(audio.to(self.device()).contiguous())
            return embed
        embeddings, timestamps = self.get_timestamp_embeddings(audio, window_size=self.max_model_window,
                                                               hop=self.scene_hop)
        return embeddings.mean(axis=1)

    def get_timestamp_embeddings(self, audio: torch.Tensor, window_size=None, hop=None, pad=None):
        """
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
        model: Loaded Model.
        Returns:
        embedding: A float32 Tensor with shape (n_sounds, n_timestamps, model.timestamp_embedding_size).
        timestamps: A float32 Tensor with shape (`n_sounds, n_timestamps). Centered timestamps in milliseconds corresponding to each embedding in the output.
        """
        if hop is None:
            hop = self.timestamp_hop
        if window_size is None:
            window_size = self.timestamp_window
        if pad is None:
            pad = window_size // 2
        audio = audio.cpu()
        n_sounds, n_samples = audio.shape
        audio = audio.unsqueeze(1)  # n_sounds,1, (n_samples+pad*2)
        # print(audio.shape)
        padded = F.pad(audio, (pad, pad), mode='reflect')
        # print(padded.shape)
        padded = padded.unsqueeze(1)  # n_sounds,1, (n_samples+pad*2)
        # print(padded.shape)
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
