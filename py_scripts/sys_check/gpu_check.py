import torch


def check_gpu():
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if not cuda_available:
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {gpu_name}")

    try:
        x = torch.rand(1000, 1000).to('cuda')
        y = torch.rand(1000, 1000).to('cuda')

        result = torch.matmul(x, y)
        print("GPU operation test passed: Tensor multiplication successful.")
    except Exception as e:
        print(f"GPU operation test failed: {e}")

    try:
        x_cpu = x.cpu()
        print("GPU to CPU tensor transfer test passed.")
    except Exception as e:
        print(f"GPU to CPU tensor transfer test failed: {e}")

    if torch.cuda.device_count() > 1:
        try:
            x_gpu1 = torch.rand(1000, 1000).to('cuda:0')
            x_gpu2 = torch.rand(1000, 1000).to('cuda:1')
            result = torch.matmul(x_gpu1, x_gpu2)
            print("Multi-GPU operation test passed.")
        except Exception as e:
            print(f"Multi-GPU operation test failed: {e}")
    else:
        print("Multi-GPU operation test skipped: Only 1 GPU available.")

    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    cached_memory = torch.cuda.memory_reserved(0)
    print(f"Total memory: {total_memory / 1e9:.2f} GB")
    print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
    print(f"Cached memory: {cached_memory / 1e9:.2f} GB")


if __name__ == "__main__":
    check_gpu()