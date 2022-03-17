import os
import sys
from inject_locals import inject

folder = sys.argv[1]
for root, dirs, files in os.walk(folder):
    for name in files:
        inject(os.path.join(root, name))
