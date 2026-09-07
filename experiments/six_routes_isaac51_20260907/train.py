"""统一训练入口：用--variant选择论文路线，默认10000轮。"""
from estnet.run import main

if __name__ == "__main__":
    raise SystemExit(main(mode_override="train"))
