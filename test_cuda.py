import torch
import time
import os

def print_separator(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

print_separator("CUDA基本信息")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    
    # 显示环境变量
    print_separator("环境变量")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
    print(f"KMP_DUPLICATE_LIB_OK: {os.environ.get('KMP_DUPLICATE_LIB_OK', '未设置')}")
    
    # 测试GPU内存
    print_separator("GPU内存信息")
    print(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
    print(f"已分配内存: {torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024:.2f} GB")
    print(f"缓存内存: {torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024:.2f} GB")
    
    # 基本GPU操作测试
    print_separator("基本GPU操作测试")
    
    # 创建CPU张量
    print("创建CPU张量...")
    cpu_tensor = torch.rand(1000, 1000)
    
    # 测试CPU到GPU的传输
    print("将张量从CPU传输到GPU...")
    start_time = time.time()
    gpu_tensor = cpu_tensor.cuda()
    transfer_time = time.time() - start_time
    print(f"传输时间: {transfer_time:.6f} 秒")
    print(f"张量设备: {gpu_tensor.device}")
    
    # 测试GPU上的矩阵乘法
    print("\n测试GPU上的矩阵乘法...")
    a = torch.rand(2000, 2000).cuda()
    b = torch.rand(2000, 2000).cuda()
    
    # 预热
    for _ in range(5):
        _ = torch.matmul(a, b)
    
    # 计时
    start_time = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()  # 确保GPU操作完成
    gpu_time = time.time() - start_time
    print(f"GPU矩阵乘法时间: {gpu_time:.6f} 秒")
    
    # 在CPU上进行相同的操作进行比较
    print("\n测试CPU上的矩阵乘法...")
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    
    # 预热
    for _ in range(5):
        _ = torch.matmul(a_cpu, b_cpu)
    
    # 计时
    start_time = time.time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU矩阵乘法时间: {cpu_time:.6f} 秒")
    
    # 比较结果
    print(f"\nGPU比CPU快: {cpu_time / gpu_time:.2f}倍")
    
    # 验证结果正确性
    diff = torch.abs(c_cpu - c.cpu()).max().item()
    print(f"CPU和GPU结果最大差异: {diff}")
    print(f"结果是否一致: {diff < 1e-5}")
    
    # 测试神经网络
    print_separator("神经网络测试 - 大型网络")
    
    # 创建一个更大的神经网络
    class LargeNet(torch.nn.Module):
        def __init__(self):
            super(LargeNet, self).__init__()
            self.fc1 = torch.nn.Linear(2048, 1024)
            self.fc2 = torch.nn.Linear(1024, 512)
            self.fc3 = torch.nn.Linear(512, 256)
            self.fc4 = torch.nn.Linear(256, 128)
            self.fc5 = torch.nn.Linear(128, 10)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = self.fc5(x)
            return x
    
    # 创建模型
    print("创建大型神经网络模型...")
    model_cpu = LargeNet()
    
    # 将模型移动到GPU
    print("将模型移动到GPU...")
    model_gpu = LargeNet().cuda()
    
    # 创建大批量输入数据
    batch_size = 1024
    print(f"创建大批量输入数据 (批量大小: {batch_size})...")
    inputs = torch.rand(batch_size, 2048)
    inputs_gpu = inputs.cuda()
    
    # 预热GPU
    print("预热GPU...")
    for _ in range(10):
        _ = model_gpu(inputs_gpu)
    torch.cuda.synchronize()
    
    # 测试前向传播
    print("\n测试前向传播...")
    
    # GPU前向传播
    print("GPU前向传播 (10次迭代)...")
    start_time = time.time()
    for _ in range(10):
        outputs_gpu = model_gpu(inputs_gpu)
    torch.cuda.synchronize()
    gpu_forward_time = (time.time() - start_time) / 10
    print(f"GPU平均前向传播时间: {gpu_forward_time:.6f} 秒")
    
    # CPU前向传播
    print("CPU前向传播...")
    start_time = time.time()
    outputs_cpu = model_cpu(inputs)
    cpu_forward_time = time.time() - start_time
    print(f"CPU前向传播时间: {cpu_forward_time:.6f} 秒")
    
    print(f"\nGPU前向传播比CPU快: {cpu_forward_time / gpu_forward_time:.2f}倍")
    
    # 测试反向传播
    print("\n测试反向传播...")
    
    # 创建损失函数和优化器
    criterion = torch.nn.MSELoss()
    optimizer_gpu = torch.optim.SGD(model_gpu.parameters(), lr=0.01)
    optimizer_cpu = torch.optim.SGD(model_cpu.parameters(), lr=0.01)
    
    # 创建目标
    targets = torch.rand(batch_size, 10)
    targets_gpu = targets.cuda()
    
    # 预热GPU反向传播
    for _ in range(5):
        optimizer_gpu.zero_grad()
        loss = criterion(model_gpu(inputs_gpu), targets_gpu)
        loss.backward()
        optimizer_gpu.step()
    torch.cuda.synchronize()
    
    # GPU反向传播
    print("GPU反向传播 (10次迭代)...")
    start_time = time.time()
    for _ in range(10):
        optimizer_gpu.zero_grad()
        loss_gpu = criterion(model_gpu(inputs_gpu), targets_gpu)
        loss_gpu.backward()
        optimizer_gpu.step()
    torch.cuda.synchronize()
    gpu_backward_time = (time.time() - start_time) / 10
    print(f"GPU平均反向传播时间: {gpu_backward_time:.6f} 秒")
    
    # CPU反向传播
    print("CPU反向传播...")
    start_time = time.time()
    optimizer_cpu.zero_grad()
    loss_cpu = criterion(model_cpu(inputs), targets)
    loss_cpu.backward()
    optimizer_cpu.step()
    cpu_backward_time = time.time() - start_time
    print(f"CPU反向传播时间: {cpu_backward_time:.6f} 秒")
    
    print(f"\nGPU反向传播比CPU快: {cpu_backward_time / gpu_backward_time:.2f}倍")
    
    print_separator("测试结论")
    matrix_speedup = cpu_time / gpu_time
    forward_speedup = cpu_forward_time / gpu_forward_time
    backward_speedup = cpu_backward_time / gpu_backward_time
    
    print(f"矩阵乘法加速比: {matrix_speedup:.2f}倍")
    print(f"前向传播加速比: {forward_speedup:.2f}倍")
    print(f"反向传播加速比: {backward_speedup:.2f}倍")
    
    if matrix_speedup > 1:
        print("✅ 矩阵运算GPU加速有效")
    else:
        print("❌ 矩阵运算GPU加速无效")
        
    if forward_speedup > 1:
        print("✅ 神经网络前向传播GPU加速有效")
    else:
        print("❌ 神经网络前向传播GPU加速无效")
        
    if backward_speedup > 1:
        print("✅ 神经网络反向传播GPU加速有效")
    else:
        print("❌ 神经网络反向传播GPU加速无效")
        
    print("\n总体结论:")
    if matrix_speedup > 1 and forward_speedup > 1 and backward_speedup > 1:
        print("✅ CUDA正常工作，GPU加速有效")
    elif matrix_speedup > 1:
        print("⚠️ CUDA部分工作，矩阵运算有加速但神经网络训练可能存在问题")
        print("   这可能是因为网络或批量太小，GPU初始化开销超过了计算收益")
        print("   在实际大型模型训练中，GPU仍应提供显著加速")
    else:
        print("❌ CUDA可能存在严重问题，GPU未能提供任何加速")
else:
    print("\n当前环境中没有可用的GPU")
    print("请检查以下可能的原因:")
    print("1. 系统中没有NVIDIA GPU")
    print("2. NVIDIA驱动程序未安装或版本不兼容")
    print("3. CUDA工具包未安装或版本不兼容")
    print("4. PyTorch安装时未启用CUDA支持")
