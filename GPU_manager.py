# import os
# import subprocess
# import time
#
# def assign_free_gpus(threshold_vram_usage=1500, max_gpus=2, wait=False, sleep_time=10):
#     """
#     Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable
#     This function should be called after all imports,
#     in case you are setting CUDA_AVAILABLE_DEVICES elsewhere
#     Borrowed and fixed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696
#     Args:
#         threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
#                                               Defaults to 1500 (MiB).
#         max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
#                                   Defaults to 2.
#         wait (bool, optional): Whether to wait until a GPU is free. Default False.
#         sleep_time (int, optional): Sleep time (in seconds) to wait before checking GPUs, if wait=True. Default 10.
#     """
#
#     def _check():
#         # Get the list of GPUs via nvidia-smi
#         smi_query_result = subprocess.check_output(
#             "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
#         )
#         # Extract the usage information
#         gpu_info = smi_query_result.decode("utf-8").split("\n")
#         gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
#         gpu_info = [
#             int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
#         ]  # Remove garbage
#         # Keep gpus under threshold only
#         free_gpus = [
#             str(i) for i, mem in enumerate(gpu_info) if mem < threshold_vram_usage
#         ]
#         free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
#         gpus_to_use = ",".join(free_gpus)
#         return gpus_to_use
#
#     while True:
#         gpus_to_use = _check()
#         if gpus_to_use or not wait:
#             break
#         print(f"No free GPUs found, retrying in {sleep_time}s")
#         time.sleep(sleep_time)
#
#     if not gpus_to_use:
#         raise RuntimeError("No free GPUs found")
#     os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_use
#     logger.info(f"Using GPU(s): {gpus_to_use}")

import subprocess
import sys
# if sys.version_info[0] < 3:
#     from StringIO import StringIO
# else:
#     from io import StringIO
import io
import pandas as pd
import time
import random
def GPU_status():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df  =pd.read_csv(io.BytesIO(gpu_stats),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    return gpu_df

def get_free_gpu(threshold=10000, sleep = 30, targets=None, check=60, every=1):
    gpu_df = GPU_status()
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))

    indexes = gpu_df.index[gpu_df['memory.free']>threshold].tolist()
    if targets!=None:
        indexes =list(set(targets).intersection(set(indexes)))
    random.shuffle(indexes)

    if len(indexes)>0 :
        print(" GPUs :"+str(indexes)+"sounds available")
        for i in range(0,check):
            print(str(i)+"th check")
            time.sleep(every)
            gpu_df = GPU_status()
            gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))

            DC_indexes = gpu_df.index[gpu_df['memory.free'] > threshold].tolist()
            indexes = list(set(DC_indexes).intersection(set(indexes)))
            if len(indexes)==0:
                break
        if  len(indexes)>0: return indexes
        else: pass
    if sleep!=None:
            print(f"No free GPUs found, retrying in {sleep}s")
            time.sleep(sleep)
            return get_free_gpu(threshold,sleep,targets)
    else:
        print("available GPUs are: "+ str(indexes))
        return indexes

# free_gpu_id = get_free_gpu()
# torch.cuda.set_device(free_gpu_id)
if __name__ == "__main__":
    print(get_free_gpu(28000,10,[0],60,1))
