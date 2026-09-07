"""拒绝更新的rollout不能被误当成已经执行Adam；检查计数与存档一致。"""
import copy
from dataclasses import replace
import pytest
import torch
from estnet.config import Config
from estnet.networks import EstNet
from estnet.ppo import PPO
from estnet.resume import _load_and_validate_optimizer


def setup_model():
    cfg=replace(Config(),encoder_hidden=(8,),actor_hidden=(8,),critic_hidden=(8,))
    m=EstNet(cfg)
    return m,PPO(m,cfg)


def test_completed_rollout_with_no_accepted_step_can_restore_empty_adam():
    m,p=setup_model();p.updates=1
    wrapped=copy.deepcopy(p.state_dict())
    assert wrapped['total_accepted_steps']==0 and wrapped['optimizer']['state']=={}
    n,q=setup_model()
    _load_and_validate_optimizer(n,q,{'iteration':1,'optimizer':wrapped})
    assert q.updates==1 and q.total_accepted_steps==0 and not q.optimizer.state


def test_accepted_count_matches_all_adam_steps_and_rejects_tampering():
    m,p=setup_model()
    sum(v.square().sum() for v in m.parameters()).backward()
    p.optimizer.step();p.updates=1;p.total_accepted_steps=1
    wrapped=copy.deepcopy(p.state_dict())
    n,q=setup_model()
    _load_and_validate_optimizer(n,q,{'iteration':1,'optimizer':wrapped})
    assert all(float(state['step'])==1 for state in q.optimizer.state.values())
    corrupt=copy.deepcopy(wrapped)
    next(iter(corrupt['optimizer']['state'].values()))['step'].fill_(2)
    with pytest.raises(ValueError,match='accepted'):
        _load_and_validate_optimizer(n,q,{'iteration':1,'optimizer':corrupt})


def test_accepted_checkpoint_cannot_silently_lose_adam_state():
    m,p=setup_model();p.updates=1
    corrupt=p.state_dict();corrupt['total_accepted_steps']=1
    with pytest.raises(ValueError,match='Adam state'):
        _load_and_validate_optimizer(m,p,{'iteration':1,'optimizer':corrupt})
