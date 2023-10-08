# 底层视觉

底层视觉问题基本涉及场景任务：超分辨率（Super resolution）、去噪（denoising）、去模糊（deblurring）、去雾去雨（dehazy）、去镜面反射（remove reflection）、水下低光照增强。前三个任务可以统一建模为下式：

$$
Y = DX + nosie
$$

其中 $Y$ 是低质量（低分辨率，噪声、模糊）图像，$X$ 是 Ground Truth，$D$ 是退化矩阵，$nosie$ 是随机干扰的加性噪声。

