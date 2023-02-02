import torch
from torch import nn
import torch.nn.functional as F
from torch import autocast
from contextlib import nullcontext


class MNHearEnsembleWrapper(nn.Module):
    def __init__(self, wrappers: List[MNHearWrapper],
                 mel: nn.Module,
                 scene_embedding_size=1487,
                 timestamp_embedding_size=1487):
        """
        @param wrappers: list of MNHearWrappers
        """
        torch.nn.Module.__init__(self)
        self.mel = mel
        self.device_proxy = nn.Parameter(torch.zeros((1)))
        self.sample_rate = mel.sr  # required to be specified according to HEAR API
        self.scene_embedding_size = scene_embedding_size  # required by HEAR API
        self.timestamp_embedding_size = timestamp_embedding_size  # required by HEAR API

    def device(self):
        return self.device_proxy.device

    def get_scene_embeddings(self, audio):
        """
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
        model: Loaded Model.
        Returns:
        embedding: A float32 Tensor with shape (n_sounds, model.scene_embedding_size).
        """
        return torch.cat([model.get_scene_embeddings(audio) for model in wrapper])

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
