import torch

from hear_mn.models.MobileNetV3 import get_model
from hear_mn.models.preprocess import AugmentMelSTFT
from hear_mn.hear_wrapper import MNHearWrapper
from hear_mn.helpers.utils import NAME_TO_WIDTH


def load_model(model_file_path="", model_name="mn10_as", mode=("b4", "b5", "b6", "b7"), all_blocks=True):
    model = get_basic_model(model_name=model_name, mode=mode, all_blocks=all_blocks)
    if torch.cuda.is_available():
        model.cuda()
    return model


def get_scene_embeddings(audio, model):
    """
    audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
    model: Loaded Model.
    Returns:
    embedding: A float32 Tensor with shape (n_sounds, model.scene_embedding_size).
    """
    model.eval()
    with torch.no_grad():
        return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    """
    audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
    model: Loaded Model.
    Returns:
    embedding: A float32 Tensor with shape (n_sounds, n_timestamps, model.timestamp_embedding_size).
    timestamps: A float32 Tensor with shape (`n_sounds, n_timestamps). Centered timestamps in milliseconds corresponding to each embedding in the output.
    """
    model.eval()
    with torch.no_grad():
        return model.get_timestamp_embeddings(audio)


def get_basic_model(model_name="mn10_as", **kwargs):
    mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024)
    net = get_model(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name,
                    collect_component_ids=tuple(list(range(16))))
    model = MNHearWrapper(mel=mel, net=net, scene_embedding_size=200, timestamp_embedding_size=200, **kwargs)
    return model
