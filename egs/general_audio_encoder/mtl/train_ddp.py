import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

def setup_distributed():
    """Setup distributed training environment."""
    dist.init_process_group(
        backend="nccl",  # 推荐 NCCL 后端
        init_method="env://",  # 使用环境变量设置
    )
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)  # 当前进程绑定到对应的 GPU

def cleanup_distributed():
    """Clean up distributed training environment."""
    dist.destroy_process_group()

def main():
    setup_distributed()
    
    # 超参数
    epochs = 5
    batch_size = 32
    lr = 0.001
    
    # 创建一个简单的数据集
    dataset = TensorDataset(torch.randn(1000, 10), torch.randn(1000, 1))
    sampler = DistributedSampler(dataset)  # 使用 DistributedSampler
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    
    # 创建一个简单的模型
    model = nn.Linear(10, 1).to()
    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
    
    # 优化器和损失函数
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 训练循环
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # 确保每个 epoch 的数据分布不同
        model.train()
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Rank {dist.get_rank()}, Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}")
    
    cleanup_distributed()

if __name__ == "__main__":
    main()