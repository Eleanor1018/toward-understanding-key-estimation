# Key1 / Key2：G1平地500轮试训

依据用户提供的 arXiv:2403.05868v1，第2页图2、第5页III.C实验组定义、第8页表III。
本轮从零训练Key1、Key2各500次外层iteration，每批仍进行4次PPO epochs；不加载EstNet权重。
正在4090-6运行的EstNet10000轮续训使用已冻结源码，不受本次本地扩展影响。

## 结构与张量契约

| 项目 | Key1 | Key2 |
|---|---|---|
| 历史输入 | 过去50帧本体观测，50×42 | 同左 |
| 显式估计 | base线速度3D | base线速度3D＋足周高度图18D |
| 隐式编码 | μ/logvar各16D | μ/logvar各16D |
| Actor输入 | 当前观测42＋命令7＋速度估计3＋μ16＝68 | 另外加入预测高度图18，共86 |
| Decoder输入 | 重参数化z16＋速度估计3＝19 | 另外加入预测高度图18，共37 |
| Decoder输出 | 当前42D本体观测 | 同左 |
| Critic输入 | 当前清洁观测42＋命令7＋特权103＝152 | 同左 |

特权103在本实现中为：真实速度3、base高度1、足周图18、base周围图81。它们只进入critic和监督标签，
actor从历史自行估计。actor/critic隐藏层为[2048,512,128]，encoder为[1024,256,64]，ELU。
KeyPolicy保持原始Gaussian动作/概率接口，动作尺度、软限位、PD、步态与奖励沿用现有G1平地基线。

历史在正常t步排除当前o_t；先取历史快照再写入当前帧。重置时只用新回合首帧填充缺失历史。
当前帧重建目标与速度/高度图标签均在env.step前复制；终帧critic在自动reset前保存。

## 论文明确与本次约定

论文明确Key1速度＋16维latent，Key2速度＋足周高度图；Key2 bullet没有重写latent维数，沿用16是共同框架推断。
表III明确velocity系数1、heightmap系数0.5、prediction系数2、VAE β50；body-height系数2不用于这两个显式输出。

以下是论文未给全或存在冲突、必须标明的选择：

1. 图2 encoder输入写o_{t−1:t−h}，decoder输出却标ô_{t+1}；图注写重建current、重建虚线连接o_t。
   本次按图注和虚线实现“过去50帧→当前o_t”，不能把此选择说成论文毫无歧义的定义。
2. 论文未给decoder隐藏宽度，本次取[64,256,1024]；critic取与actor相同宽度。
3. 论文未给latent概率头、采样/部署约定或detach。共享trunk后接显式量、μ与logvar heads；
   actor始终用μ，训练decoder用重参数化采样，评估用μ。PPO、监督与重建共同更新encoder，不detach。
   actor使用确定性μ避免PPO重算概率时因latent重采样改变条件分布；这不等于复原作者未公开代码。
4. 辅助loss取 `1*MSE_velocity + 0.5*MSE_heightmap(Key2 only) + 2*MSE_prediction + 50*KL`。
   MSE对batch和全部特征取mean，KL对batch和latent维同时取mean。系数来自表III；归约方式及β与prediction系数
   的组合方式为工程约定。logvar裁剪[-10,10]仅作数值保护。PPO/value/entropy项保留原基线。
5. 足图每脚3×3，x偏移[-0.1,0,0.1]m、y偏移[-0.05,0,0.05]m，先左9再右9；base图9×9，
   x/y范围[-0.4,0.4]m、间隔0.1m。只绕base yaw旋转，输出采样原点到地面的世界竖直距离（米）。
   足图参考对应脚踝link高度，base图参考base高度，不裁剪、不归一化。论文未给这些网格/坐标/参考高度；
   18＋81并不是能从P103唯一推得的作者设置。

## 初试的范围和比较限制

先做水平平地，关闭噪声、延迟、域随机化，前向命令0.25–0.55m/s；G1、IsaacLab2.0.2/IsaacSim4.5，
100Hz策略、1kHz物理、12腿动作、上身17关节PD。论文使用Wukong-IV、IsaacGym和更广的训练分布，
因此此次是G1上的结构复现与启动验证，尚不复现复杂地形鲁棒性。

对水平平面，地表高度用精确解析交点计算，足图并非硬编码为零；抬脚会改变对应足图距离。
同一个脚图在平地的9个点相同，大图81点也相同，这是当前地形的退化性质。
更换坡面/粗糙地形必须接入真正逐点地面查询，不能继续用平面常数。

Key1与Key2共用同一环境、critic、奖励和随机种子，主要差异为显式高度图估计。
先前EstNet使用61D critic与含当前帧的50帧历史；为了保留其已启动续训，本次未改变这两个历史约定。
因此现有EstNet日志与新Key系列不是严格只改变latent结构的三组消融；严谨三组对比须另开统一观察/critic的EstNet实验。

## 启动与验证

```bash
python -m estnet.run train --variant key1 --asset "$G1_USD" --iterations 500 --num-envs 4096 --headless
python -m estnet.run train --variant key2 --asset "$G1_USD" --iterations 500 --num-envs 4096 --headless
python -m estnet.run evaluate --checkpoint /absolute/key2/model_00500.pt --asset "$G1_USD" --num-envs 32 --headless
```

各variant有独立schema；评估和续训从检查点选择结构，显式指定不同variant会在CPU预检拒绝。
网络和训练器工厂用于训练、评估、checkpoint验证和Adam恢复，旧EstNet检查点仍可加载。
每100轮保存；训练结束仅触发独立均值动作首回合评估，不把进程退出、奖励上升或辅助MSE降低当作行走成功。
