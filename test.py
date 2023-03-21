import torch
from hear_mn import mn01_all_b_mel_avgs

seconds = 20
sampling_rate = 32000
audio = torch.ones((1, sampling_rate * seconds))
wrapper = mn01_all_b_mel_avgs.load_model().cuda()

embed, time_stamps = wrapper.get_timestamp_embeddings(audio)
print(embed.shape)
embed = wrapper.get_scene_embeddings(audio)
print(embed.shape)
