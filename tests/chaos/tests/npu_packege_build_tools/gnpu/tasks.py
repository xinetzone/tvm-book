from invoke import task 

@task
def module(ctx, name="vta", output_dir="../"):
    cmd = "cd src && python3 -m nuitka --remove-output "
    cmd += f"--no-pyi-file --output-dir={output_dir} "
    cmd += f"--module {name} --include-package={name} "
    ctx.run(cmd)

@task
def group(ctx, name="vta", output_dir="../"):
    cmd = "cd src && python3 -m nuitka --remove-output "
    cmd += f"--no-pyi-file --output-dir={output_dir} "
    cmd += f"--module {name} --include-package={name} "
    ctx.run(cmd)