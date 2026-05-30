"""
对比 cls / mean / max 三种池化（等价于 evaluate.py --all_pools）

  python compare_pools.py
  python compare_pools.py --fast
"""

import sys

if __name__ == "__main__":
    if "--all_pools" not in sys.argv:
        sys.argv.insert(1, "--all_pools")
    from evaluate import main
    main()
