import os
import sys
import shutil
import subprocess
from pathlib import Path

def run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
    except Exception as e:
        return f"ERROR: {e}"

def find_cuda_candidates():
    # 常见 CUDA 安装位置
    candidates = [
        "/usr/local/cuda",
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-12.7",
        "/usr/local/cuda-12.6",
        "/opt/cuda",
    ]
    # Conda 环境里的可能路径
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates += [
            str(Path(conda_prefix) / "pkgs" / "cuda-toolkit"),
            str(Path(conda_prefix)),
        ]
    return [p for p in candidates if Path(p).exists()]

def parse_pytorch_cuda():
    try:
        import torch
        is_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if is_avail else None
        # 解析 PyTorch 构建的 CUDA 版本线索，如 cu128/cu121 等
        b = torch.version.cuda  # e.g. '12.1' or None for CPU builds
        # 再从 torch.version 分发标签中获得更详细的线索
        dist = torch.version.__dict__
        return {
            "torch_version": torch.__version__,
            "torch_cuda_version": b,  # '12.1'/'12.4'/None
            "torch_cuda_available": is_avail,
            "gpu_name": device_name,
        }
    except Exception as e:
        return {"error": str(e)}

def get_nvcc_info():
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return {"nvcc_found": False}
    out = run([nvcc, "--version"])
    return {"nvcc_found": True, "nvcc_path": nvcc, "nvcc_version": out}

def try_set_cuda_env(cuda_home):
    if not cuda_home:
        return False
    cuda_home = str(cuda_home)
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["PATH"] = f"{cuda_home}/bin:" + os.environ.get("PATH", "")
    lib64 = f"{cuda_home}/lib64"
    os.environ["LD_LIBRARY_PATH"] = (lib64 + ":" + os.environ.get("LD_LIBRARY_PATH", "")) if Path(lib64).exists() else os.environ.get("LD_LIBRARY_PATH", "")
    return True

def header_exists(cuda_home=None):
    # 检查 cuda_runtime_api.h
    candidates = []
    if cuda_home:
        candidates.append(Path(cuda_home) / "include" / "cuda_runtime_api.h")
    candidates += [
        Path("/usr/local/cuda/include/cuda_runtime_api.h"),
        Path("/opt/cuda/include/cuda_runtime_api.h"),
    ]
    for p in candidates:
        if p.exists():
            return True, str(p)
    return False, None

def suggest_install_commands(torch_cuda_version):
    cmds = []
    # 优先建议 conda 安装（版本匹配更容易）
    if torch_cuda_version:
        major_minor = torch_cuda_version
        cmds.append({
            "label": f"Conda 安装与 PyTorch 匹配的 CUDA Toolkit ({major_minor})",
            "cmd": f"conda install -c nvidia cuda-toolkit"
        })
    else:
        cmds.append({
            "label": "Conda 安装 CUDA Toolkit（通用）",
            "cmd": "conda install -c nvidia cuda-toolkit"
        })
    # Ubuntu 系系统包（可能版本较旧）
    cmds.append({
        "label": "Ubuntu/Debian 安装系统 CUDA（可能版本不匹配）",
        "cmd": "sudo apt update && sudo apt install -y nvidia-cuda-toolkit"
    })
    cmds.append({
        "label": "NVIDIA 官网安装（选择与 PyTorch 构建匹配的版本）",
        "cmd": "打开 https://developer.nvidia.com/cuda-downloads 手动安装对应版本"
    })
    return cmds

def clear_torch_extensions_cache():
    cache_dir = Path.home() / ".cache" / "torch_extensions"
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            return True, str(cache_dir)
        except Exception as e:
            return False, f"删除失败: {e}"
    return None, str(cache_dir)

def set_cuda(auto_fix=True, clear_cache=False):
    print("=== CUDA 自动检查与修复 ===")

    # 1. PyTorch 与 CUDA 线索
    info = parse_pytorch_cuda()
    if "error" in info:
        print(f"- PyTorch 检查: 失败 ({info['error']})")
    else:
        print(f"- PyTorch 版本: {info['torch_version']}")
        print(f"- PyTorch CUDA 可用: {info['torch_cuda_available']}")
        if info['gpu_name']:
            print(f"- GPU: {info['gpu_name']}")
        print(f"- PyTorch 构建 CUDA 版本线索: {info['torch_cuda_version']}")

    # 2. nvcc 检查
    nvcc = get_nvcc_info()
    if nvcc.get("nvcc_found"):
        print(f"- 已检测到 nvcc: {nvcc['nvcc_path']}")
        print(nvcc['nvcc_version'])
    else:
        print("- 未检测到 nvcc（CUDA Toolkit 可能未安装或 PATH 未包含 nvcc）")

    # 3. CUDA_HOME 与头文件
    current_cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    print(f"- 当前 CUDA_HOME: {current_cuda_home}")
    exists, header_path = header_exists(current_cuda_home)
    if exists:
        print(f"- 已找到头文件 cuda_runtime_api.h: {header_path}")
    else:
        print("- 未找到 cuda_runtime_api.h 头文件")

    # 4. 尝试自动设置 CUDA_HOME
    success_set = False
    chosen_home = None
    if auto_fix:
        if not current_cuda_home or not exists:
            candidates = find_cuda_candidates()
            # 优先选择包含 include/cuda_runtime_api.h 的路径
            for c in candidates:
                ok, p = header_exists(c)
                if ok:
                    chosen_home = c
                    break
            # 如果候选不含头文件，仍尝试设置第一个存在的路径
            if not chosen_home and candidates:
                chosen_home = candidates[0]

            if chosen_home:
                success_set = try_set_cuda_env(chosen_home)
                exists2, header_path2 = header_exists(chosen_home)
                print(f"- 自动设定 CUDA_HOME={chosen_home} {'成功' if success_set else '失败'}")
                if exists2:
                    print(f"- 头文件检测通过: {header_path2}")
            else:
                print("- 未找到可用的 CUDA 安装路径候选，无法自动设定 CUDA_HOME")

    # 5. 再次汇总环境
    print(f"- 最终 CUDA_HOME: {os.environ.get('CUDA_HOME')}")
    print(f"- PATH 中是否包含 nvcc: {'nvcc' in run(['which','nvcc']) if shutil.which('which') else '未知'}")

    # 6. 修复建议
    print("\n=== 修复建议 ===")
    need_toolkit = (not nvcc.get("nvcc_found")) or (not header_exists(os.environ.get("CUDA_HOME"))[0])
    if need_toolkit:
        print("- 检测到 CUDA 开发环境不完整（缺少 nvcc 或头文件）。建议安装 CUDA Toolkit：")
        for item in suggest_install_commands(info.get("torch_cuda_version")):
            print(f"  * {item['label']}:")
            print(f"    {item['cmd']}")
    else:
        print("- 已检测到 CUDA Toolkit 与头文件。若仍编译失败，检查 CUDA 与 PyTorch 版本是否匹配。")

    # 7. 可选清理 PyTorch 扩展缓存
    if clear_cache:
        print("\n=== 清理 PyTorch 扩展缓存 ===")
        ok, msg = clear_torch_extensions_cache()
        if ok is True:
            print(f"- 已清理缓存目录: {msg}")
        elif ok is False:
            print(f"- 清理失败: {msg}")
        else:
            print(f"- 缓存目录不存在: {msg}")

    # 8. 额外提示
    print("\n=== 额外提示 ===")
    print("- 在 Jupyter 中设置的环境变量仅对当前内核有效。若要永久生效，请把 CUDA_HOME/PATH/LD_LIBRARY_PATH 写入 ~/.bashrc 或 conda 激活脚本。")
    print("- 若 PyTorch 是 CPU 版本或与 CUDA 版本不匹配，编译扩展会失败。可改装 PyTorch 的 CUDA 版本或使用匹配的 Toolkit。")
    print("- 如果系统装了多个 CUDA 版本，请确保 PATH 与 LD_LIBRARY_PATH 指向与 PyTorch 构建匹配的版本。")

# 在 Notebook 里运行：
# 1) 只检查并自动修复（不清缓存）
# set_cuda(auto_fix=True, clear_cache=False)
# 2) 检查、自动修复并清理扩展缓存
# set_cuda(auto_fix=True, clear_cache=True)

set_cuda(auto_fix=True, clear_cache=True)
