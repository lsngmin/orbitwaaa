"""pytest sys.path 설정.

`submission/` 안의 모듈 (submission_features, submission_actor, main) 을
테스트가 직접 import 하므로 repo root + submission/ 둘 다 sys.path 에 둔다.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.join(_HERE, "submission")):
    if p not in sys.path:
        sys.path.insert(0, p)
