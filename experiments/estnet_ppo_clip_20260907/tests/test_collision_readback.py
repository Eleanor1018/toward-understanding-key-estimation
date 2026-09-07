"""只检验运行时属性读取/拒绝逻辑，不把替代 USD 对象称为物理接触测试。"""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from estnet.run import check_live_self_collision


def install_stage(monkeypatch, values):
    class Prim:
        def __init__(self, value):
            self.value = value

        def IsValid(self):
            return True

        def GetAttribute(self, name):
            assert name == "physxArticulation:enabledSelfCollisions"
            return SimpleNamespace(IsValid=lambda: self.value is not None, Get=lambda: self.value)

        def GetPath(self):
            return "/World/envs/env_0/Robot/root"

    robot = Prim(True)

    def get_prim(path):
        assert path == "/World/envs/env_0/Robot"
        return robot

    stage = SimpleNamespace(GetPrimAtPath=get_prim)
    omni = ModuleType("omni")
    usd = ModuleType("omni.usd")
    usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
    omni.usd = usd
    pxr = ModuleType("pxr")
    pxr.Usd = SimpleNamespace(PrimRange=lambda root: [Prim(value) for value in values])
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.usd", usd)
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    return SimpleNamespace(robot=SimpleNamespace(spawn=SimpleNamespace(
        articulation_props=SimpleNamespace(enabled_self_collisions=True))))


def test_composed_true_attribute_is_recorded_without_claiming_contact_response(monkeypatch):
    cfg = install_stage(monkeypatch, [None, True])
    report = check_live_self_collision(cfg)
    assert report["attributes"] == [{"prim": "/World/envs/env_0/Robot/root", "enabled": True}]
    assert report["contact_response_verified"] is False


@pytest.mark.parametrize("values", [[], [None], [False], [True, False]])
def test_missing_or_disabled_live_attribute_refuses_training(monkeypatch, values):
    cfg = install_stage(monkeypatch, values)
    with pytest.raises(RuntimeError, match="Self collision"):
        check_live_self_collision(cfg)
