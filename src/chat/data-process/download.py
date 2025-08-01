#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('mouseEliauk/mirau-7b-RP-base', cache_dir='../../../gal-chat/')