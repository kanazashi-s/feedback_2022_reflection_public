import torch


def check_gpu():
    if torch.cuda.is_available():
        print("GPU is available")
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))
    else:
        print("GPU not available")


if __name__ == '__main__':
    check_gpu()