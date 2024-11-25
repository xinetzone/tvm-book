from pathlib import Path
import asyncio
from invoke import task
from d2py.utils.file import mkdir
from tqdm.asyncio import tqdm

ROOT = Path("../tasks").resolve()
# ROOT = Path("/home/workspace").resolve()
TEMP = ROOT/"temp-new"
ORIGIN = TEMP/"tvm"
TARGET = TEMP/"gnpu"

@task
def cpp(ctx):
    ctx.run(f"rm -rf {TEMP}")
    mkdir(ORIGIN)
    mkdir(f'{TARGET}/src/bin')
    mkdir(f'{TARGET}/src/tvm')
    mkdir(f'{TARGET}/src/tvm/relay')
    mkdir(f'{TARGET}/src/tvm/relay/quantize')
    mkdir(f"{TARGET}/src/vta_hw/build")
    ctx.run(f"cp -rf /media/pc/data/lxw/ai/tvm/* {ORIGIN}")
    ctx.run(f"cp -rf xm-gnpu/* {TARGET}")
    ctx.run(f"cp -rf {ORIGIN}/build/libtvm*.so {TARGET}/src/bin")
    ctx.run(f"cp {ORIGIN}/build/libvta*.so {TARGET}/src/vta_hw/build")
    ctx.run(f"cp {TARGET}/replace/vta/environment.py {ORIGIN}/vta/python/vta")
    ctx.run(f"cp -rf {ORIGIN}/3rdparty/vta-hw/config/ {TARGET}/src/vta_hw")

@task
async def package(ctx,
            name="vta",
            origin_dir="vta/python",
            out_dir="src", 
            is_package=True):
    cmd = f"cd {ORIGIN}/{origin_dir} && "
    cmd += "python3 -m nuitka --remove-output --no-pyi-file "
    cmd += f"--output-dir={TARGET}/{out_dir} "
    cmd += f"--module {name} "
    if is_package:
        cmd += f"--include-package={name}"
    ctx.run(cmd)

@task
async def tvm(ctx,
        origin_dir="python/tvm",
        out_dir="src/tvm"):
    root = ORIGIN/origin_dir
    for path in tqdm(root.iterdir()):
        name = path.name
        if name.endswith(".pyi"):
            continue
        if name not in ["__pycache__", "relay"]:
            if name == "__init__.py":
                ctx.run(f"cp -rf {path} {TARGET}/{out_dir}/")
                continue
            if path.is_dir():
                is_package = True
            else:
                is_package = False
            await package(ctx,
                    name=name,
                    origin_dir=origin_dir,
                    out_dir=out_dir, 
                    is_package=is_package)

@task     
async def relay(ctx,
          origin_dir="python/tvm/relay",
          out_dir="src/tvm/relay"):
    root = ORIGIN/origin_dir
    for path in tqdm(root.iterdir()):
        print(path)
        name = path.name
        if name.endswith(".pyi"):
            continue
        if name not in ["__pycache__", "quantize"]:
            if name in ["__init__.py", "std"]:
                ctx.run(f"cp -rf {path} {TARGET}/{out_dir}/")
                continue
            if path.is_dir():
                is_package = True
            else:
                is_package = False
            await package(ctx,
                    name=name,
                    origin_dir=origin_dir,
                    out_dir=out_dir, 
                    is_package=is_package)
            
@task     
async def quantize(ctx,
             origin_dir="python/tvm/relay/quantize",
             out_dir="src/tvm/relay/quantize"):
    root = ORIGIN/origin_dir
    for path in tqdm(root.iterdir()):
        print(path)
        name = path.name
        if name.endswith(".pyi"):
            continue
        if name not in ["__pycache__"]:
            if name in ["__init__.py", "_calibrate.py"]:
                ctx.run(f"cp -rf {path} {TARGET}/{out_dir}/")
                continue
            if path.is_dir():
                is_package = True
            else:
                is_package = False
            await package(ctx,
                    name=name,
                    origin_dir=origin_dir,
                    out_dir=out_dir, 
                    is_package=is_package)

@task     
async def pack(ctx):
    # VTA
    await asyncio.create_task(package(ctx,
            name="vta",
            origin_dir="vta/python",
            out_dir="src", 
            is_package=True))
    # TVM
    await asyncio.gather(
        tvm(ctx,
            origin_dir="python/tvm",
            out_dir="src/tvm"),
        relay(ctx,
            origin_dir="python/tvm/relay",
            out_dir="src/tvm/relay"),
        quantize(ctx,
            origin_dir="python/tvm/relay/quantize",
            out_dir="src/tvm/relay/quantize")
    )
    
@task
def all(ctx):
    cpp(ctx) # 准备工作
    # 打包
    asyncio.run(pack(ctx))
    # 打包
    ctx.run(f"cd {TARGET};hatch build")