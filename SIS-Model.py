# 1. SIS 模型，常微分方程，解析解与数值解的比较
from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np  # 导入 numpy包
import matplotlib.pyplot as plt  # 导入 matplotlib包

# ​
def dy_dt(y, t, lamda, mu):  # SIS 模型，导数函数
    dy_dt = lamda*y*(1-y) - mu*y  # di/dt = lamda*i*(1-i)-mu*i
    return dy_dt

# 设置模型参数
number = 1e5  # 总人数
lamda = 1.2  # 日接触率, 患病者每天有效接触的易感者的平均人数
sigma = 2.5  # 传染期接触数
mu = lamda/sigma  # 日治愈率, 每天被治愈的患病者人数占患病者总数的比例
fsig = 1-1/sigma
y0 = i0 = 1e-5  # 患病者比例的初值
tEnd = 50  # 预测日期长度
t = np.arange(0.0,tEnd,1)  # (start,stop,step)
print("lamda={}\tmu={}\tsigma={}\t(1-1/sig)={}".format(lamda,mu,sigma,fsig))

# 解析解
if lamda == mu:
    yAnaly = 1.0/(lamda*t +1.0/i0)
else:
    yAnaly= 1.0/((lamda/(lamda-mu)) + ((1/i0)-(lamda/(lamda-mu))) * np.exp(-(lamda-mu)*t))
# odeint 数值解，求解微分方程初值问题
ySI = odeint(dy_dt, y0, t, args=(lamda,0))  # SI 模型
ySIS = odeint(dy_dt, y0, t, args=(lamda,mu))  # SIS 模型

# 绘图
plt.plot(t, yAnaly, '-ob', label='analytic')
plt.plot(t, ySIS, ':.r', label='ySIS')
plt.plot(t, ySI, '-g', label='ySI')

plt.title("Comparison between analytic and numerical solutions")
plt.axhline(y=fsig,ls="--",c='c')  # 添加水平直线
plt.legend(loc='best')  # youcans
plt.axis([0, 50, -0.1, 1.1])
plt.show()