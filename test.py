import torch
from hear_mn import mn10_e1_l

seconds = 20
audio = torch.ones((1, 32000 * seconds))*0.5
wrapper = mn10_e1_l.load_model().cuda()

embed, time_stamps = wrapper.get_timestamp_embeddings(audio)
print(embed.shape)
embed = wrapper.get_scene_embeddings(audio)
print(embed.shape)
