# microTVM 参考虚拟机

**原作者**: `[Andrew Reusch](areusch@octoml.ai)

本教程解释如何启动 microTVM 参考虚拟机（Reference Virtual Machines）。您可以使用它们在真实的物理硬件上进行开发，而不需要单独安装 microTVM 依赖项。当试图重现 microTVM 的行为时，比如在提交 bug 报告时，这些也特别有用。

microTVM 允许 TVM 在 bare-metal 微控制器上构建和执行模型。microTVM 的目标是兼容各种 SoC 和运行时环境（即 bare-metal、RTOS 等）。然而，需要一些稳定的软件环境来允许开发人员共享和重现错误和结果。microTVM 参考虚拟机旨在提供这种环境。

## 它是如何工作的

没有虚拟机存储在 TVM 存储库中——相反，存储在 ``apps/microtvm/reference-vm`` 中的文件描述了如何使用 [Vagrant][Vagrant] VM 构建工具构建虚拟机。

参考虚拟机分为两部分：

1. Vagrant Base Box，它包含该平台的所有稳定依赖项。构建脚本存储在 ``apps/microtvm/reference-vm/<platform>/base-box`` 中。TVM 提交者在平台的 "stable" 依赖项发生变化时运行这些程序，生成的 Base Box 存储在 [Vagrant Cloud][Vagrant Cloud] 中。
2. 每个工作区 VM，用户通常使用 Base Box 作为起点构建它。构建脚本存储在 ``apps/microtvm/reference-vm/<platform>``（除了 ``base-box``）。

## 设置 VM

### 安装的先决条件

需要最少的先决条件集：

1. [Vagrant][Vagrant]
2. 支持的虚拟机管理程序（**VirtualBox**、**Parallels** 或 **VMWare Fusion/Workstation**）。[VirtualBox](https://www.virtualbox.org) 是建议的免费管理程序，但请注意 [VirtualBox Extension Pack](VirtualBox Extension Pack)  是正确的 USB 转发所必需的。如果使用 VirtualBox，也可以考虑安装 [vbguest](https://github.com/dotless-de/vagrant-vbguest) 插件。
3. 如果您的系统管理程序需要，可以使用 [Vagrant 提供程序插件](https://github.com/hashicorp/vagrant/wiki/Available-Vagrant-Plugins#providers)（VMWare 请参阅[此处](https://www.vagrantup.com/vmware)）。

## 首次 boot

第一次使用 RVM（reference VM） 时，您需要在本地创建该 box，然后提供它。

```bash
# Replace zephyr with the name of a different platform, if you are not using Zephyr.
~/.../tvm $ cd apps/microtvm/reference-vm/zephyr
# Replace <provider_name> with the name of the hypervisor you wish to use (i.e. virtualbox, parallels, vmware_desktop).
~/.../tvm/apps/microtvm/reference-vm/zephyr $ vagrant up --provider=<provider_name>
```


This command will take a couple of minutes to run and will require 4 to 5GB of storage on your
machine. It does the following:

1. Downloads the [microTVM base box](https://app.vagrantup.com/tlcpack/boxes/microtvm) and clones it to form a new VM specific to this TVM directory.
2. Mounts your TVM directory (and, if using ``git-subtree``, the original ``.git`` repo) into the
   VM.
3. Builds TVM and installs a Python virtualenv with the dependencies corresponding with your TVM
   build.

[Vagrant Cloud]: https://app.vagrantup.com/tlcpack

[Vagrant]: https://vagrantup.com

Connect Hardware to the VM
--------------------------

Next, you need to configure USB passthrough to attach your physical development board to the virtual
machine (rather than directly to your laptop's host OS).

It's suggested you setup a device filter, rather than doing a one-time forward, because often the
device may reboot during the programming process and you may, at that time, need to enable
forwarding again. It may not be obvious to the end user when this occurs. Instructions to do that:

- [VirtualBox](https://www.virtualbox.org/manual/ch03.html#usb-support)
- [Parallels](https://kb.parallels.com/122993)
- [VMWare Workstation](https://docs.vmware.com/en/VMware-Workstation-Pro/15.0/com.vmware.ws.using.doc/GUID-E003456F-EB94-4B53-9082-293D9617CB5A.html)


Rebuilding TVM inside the Reference VM
--------------------------------------

After the first boot, you'll need to ensure you keep the build, in ``$TVM_HOME/build-microtvm-zephyr``,
up-to-date when you modify the C++ runtime or checkout a different revision. You can either
re-provision the machine (``vagrant provision`` in the same directory you ran ``vagrant up`` before)
or manually rebuild TVM yourself.

Remember: the TVM ``.so`` built inside the VM is different from the one you may use on your host
machine. This is why it's built inside the special directory ``build-microtvm-zephyr``.

Logging in to the VM
--------------------

The VM should be available to your host only with the hostname ``microtvm``. You can SSH to the VM
as follows:

```bash
$ vagrant ssh
```

Then ``cd`` to the same path used on your host machine for TVM. For example, on Mac:


```bash
$ cd /Users/yourusername/path/to/tvm
```

Running tests
=============

Once the VM has been provisioned, tests can executed using ``poetry``:

```bash
$ cd apps/microtvm/reference-vm/zephyr
$ poetry run python3 ../../../../tests/micro/qemu/test_zephyr.py --zephyr-board=stm32f746g_disco
```

If you do not have physical hardware attached, but wish to run the tests using the
local QEMU emulator running within the VM, run the following commands instead:


```bash
$ cd /Users/yourusername/path/to/tvm
$ cd apps/microtvm/reference-vm/zephyr/
$ poetry run pytest ../../../../tests/micro/qemu/test_zephyr.py --zephyr-board=qemu_x86
```