# QuickDraw

QuickDraw 是 Google QuickDraw 的子集，用于草图分类和生成研究。

**特性**

- 表示: Polyline
- 来源: Google QuickDraw

**用法示例**

```python
from sketchkit.datasets import QuickDraw
ds = QuickDraw(root="~/.sketchkit/datasets", load_all=False)
print(ds.items_metadata.head())
```
