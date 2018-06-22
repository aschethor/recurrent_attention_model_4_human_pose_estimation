import subprocess

def get_gpu_memory_map():
    """ return gpu memory useage in MiB, code found under:discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/3 """
    result = subprocess.check_output(
        [ 'nvidia-smi','--query-gpu=memory.used','--format=csv,nounits,noheader'],
        encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)),gpu_memory))
    return gpu_memory_map
