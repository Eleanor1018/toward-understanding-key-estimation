# 五个论文变体的定义复核

依据用户提供的[论文v1](https://arxiv.org/pdf/2403.05868v1)，核对第2页图2、第4–5页III.B/C、第8页Table III，并与D盘现有变体实现比对。本报告只核定义，不表示新版五路训练已经启动或完成。

**Key1没有足周高度图估计头。** 它是速度估计＋16维隐变量；Key2才加入足周高度图。取消PPO策略KL对更新的干预时，**VAE隐变量KL的β=50仍应保留**。这是两个不同的KL。

## 每组到底估计什么、送给谁

下表维度按当前工程配置：观测42、命令7、足图18、latent16。论文没有公布足图18这一尺寸。

| 变体 | 显式监督目标 / 同时送actor的估计 | actor完整输入维度 | decoder输入 / 输出 |
|---|---|---:|---|
| Key1 | 机身线速度3 | 42+7+3+16=68 | z16+速度3=19 → obs42 |
| Key2 | 机身线速度3、足周高度图18 | 42+7+21+16=86 | z16+估计21=37 → obs42 |
| FullEst | 机身线速度3、足周高度图18、机身高度1 | 42+7+22+16=87 | z16+估计22=38 → obs42 |
| IrrEst | 仅机身高度1 | 42+7+1+16=66 | z16+高度1=17 → obs42 |
| Implicit | 无显式估计 | 42+7+16=65 | z16 → obs42 |

各组actor还输入当前观测与命令，表中的速度/高度图/高度均是encoder预测值，不能换成物理真值。decoder只读隐变量与本组显式估计，不接当前观测或标签作为输入。FullEst的“全估计”不含大范围base高度图：论文第2页II.B将大图排除在可显式估计量外，第5页也只列足周图。

论文第5页直接写出Key1、IrrEst、Implicit的latent为16维；Key2未重写维数，FullEst只说明继承第4页显著性组的固定宽度隐式信息。因此为Key2/FullEst也选16维是合理、但需声明的共同框架约定。原文名称是**IrrEst**，不是IllEst。[论文第4–5页](https://arxiv.org/pdf/2403.05868v1#page=4)。

代码核查：KeyPolicy只有`variant == key2`才建heightmap_head；所有实际显式估计均经`_explicit`送actor与decoder。[代码](D:/toward-understanding-key-estimation/estnet/key_networks.py:49) [代码](D:/toward-understanding-key-estimation/estnet/key_networks.py:77) [代码](D:/toward-understanding-key-estimation/estnet/key_networks.py:88)。旧`Config.supervision_dims`仍给Key1附带heightmap真值标签，但KeyPPO不选取它、不计算map loss；这是冗余接口，不是“有map头但不送actor”的隐藏架构。新版应删掉这一误导性冗余。[代码](D:/toward-understanding-key-estimation/estnet/config.py:80) [代码](D:/toward-understanding-key-estimation/estnet/key_ppo.py:19)

IrrEst只创建body_height_head，Implicit不创建任何显式头；不存在的速度估计返回None，不应记录伪造的velocity_loss/RMSE=0。环境的真实速度跟踪误差仍可正常记录。[代码](D:/toward-understanding-key-estimation/estnet/ablation_networks.py:50) [代码](D:/toward-understanding-key-estimation/estnet/ablation_networks.py:85) [代码](D:/toward-understanding-key-estimation/estnet/ablation_ppo.py:36)

## 重建目标：原文冲突与本轮选定时序

已目视核对图2：encoder输入写`o_(t−1:t−h)`；decoder输出标`o_hat_(t+1)`，但图注写重建current，虚线又连接当前`o_t`。这三处不能同时当成毫无歧义的“next-observation重建”。[图2原页](https://arxiv.org/pdf/2403.05868v1#page=2)。

本轮沿用旧五组约定：在动作`t`之前，历史为过去50帧、不含`o_t`；decoder预测当前`o_t`，损失对`data['obs']`计算，**没有改成`next_obs`**。显式真值也来自当前`t`：速度为机身坐标系三维线速度，高度为base世界z减平地高度，足图是当前脚采样原点到地面的竖直距离。收集器在`env.step`前clone当前观测和标签；终帧critic另在自动reset前保存，用于GAE自举，不是重建标签。[代码](D:/toward-understanding-key-estimation/estnet/history.py:14) [代码](D:/toward-understanding-key-estimation/estnet/environment.py:251) [代码](D:/toward-understanding-key-estimation/estnet/run.py:349) [代码](D:/toward-understanding-key-estimation/estnet/key_ppo.py:28)

新episode用该episode首帧重复填充尚不存在的历史，这是启动边界的工程处理；不能跨episode借旧历史。当前网络不把标签送进actor/decoder，也没有对显式估计或mu进行stop-gradient。论文未规定detach与采样细节：当前actor始终用mu，训练decoder用重参数采样，评估decoder用mu。这样PPO重新计算同一输入/动作的概率时不会因重新采样latent而改变条件分布。[代码](D:/toward-understanding-key-estimation/estnet/key_networks.py:88) [代码](D:/toward-understanding-key-estimation/estnet/ablation_networks.py:90)

## 保留的VAE与显式监督损失

论文Table III公布速度1、足图.5、机身高度2、prediction2、VAE β50；但没有完整写出VAE概率建模、MSE/KL归约与系数组合公式。[Table III](https://arxiv.org/pdf/2403.05868v1#page=8)。现有实现为以下辅助损失，另加共同PPO/value/entropy控制损失：

| 变体 | 辅助损失 |
|---|---|
| Key1 | 1·MSE速度 + 2·MSE当前观测 + 50·KL隐变量 |
| Key2 | Key1项 + .5·MSE足图 |
| FullEst | Key2项 + 2·MSE机身高度 |
| IrrEst | 2·MSE机身高度 + 2·MSE当前观测 + 50·KL隐变量 |
| Implicit | 2·MSE当前观测 + 50·KL隐变量 |

`KL隐变量 = 0.5*mean(mu²+exp(logvar)−1−logvar)`，对batch与16个latent维一起取mean；MSE也对batch与所有对应特征取mean。若改成对latent求sum再对batch平均，同样β50会放大16倍；若把整个VAE项再乘prediction系数2，也会改变当前尺度。这些归约与组合是工程约定，不能在迁移PPO基础时悄悄改变。[代码](D:/toward-understanding-key-estimation/estnet/key_ppo.py:24) [代码](D:/toward-understanding-key-estimation/estnet/ablation_ppo.py:24)

PPO策略KL比较新旧动作分布；VAE KL约束隐变量后验接近标准正态。新方案可以让前者仅监测，同时保持后者参与梯度更新。`logvar`裁剪[-10,10]、actor用mu、decoder隐藏层[64,256,1024]都是现有工程选择。encoder[1024,256,64]、backbone[2048,512,128]与ELU来自表III；独立critic沿用backbone宽度的安排仍应注明。

## 高度图与比较范围不能省略的区别

论文给出特权信息103维，但不足以唯一推出网格。当前选足图18（每脚3×3，x偏移±.1m、y±.05m）、base图81（9×9、间距.1m、范围±.4m）；网格随base yaw旋转，值为采样原点z减实际地面z、单位米。脚图以脚踝link为参考，不等于足底净空。平地解析查询会令同一脚九点相同；换粗糙地形必须查询各采样点地表，不能沿用平面常数。[代码](D:/toward-understanding-key-estimation/estnet/heightmaps.py:1)

五组critic均为`42+7+103=152`，其中103=`速度3+base高度1+足图18+base图81`。旧及正在运行的新EstNet500仍为critic61，并使用包含当前帧的50帧历史；五组使用排除当前的过去50帧。**即使统一self-collision、hip yaw、无soft-target裁剪与固定LR PPO，也不能称这些结果是“仅替换估计头”的严格消融。** 本轮保留并说明差异，不触碰正在运行的EstNet500。

旧`key_ppo.py`/`ablation_ppo.py`开头还写“自适应LR沿用基类”；这描述旧训练基础。新版文档应改为固定LR、PPO KL仅观察，同时保持上述各变体辅助损失与时序。此次审计没有修改任何训练代码，也没有启动五路GPU任务。
