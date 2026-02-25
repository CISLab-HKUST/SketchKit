# TUBerlin

TUBerlin（TU Berlin）数据集常用于草图检索与分类研究。

**特性**

- 表示: Bezier

**用法示例**

```python
from sketchkit.datasets import TUBerlin
ds = TUBerlin()
print(ds.items_metadata.category.unique()[:10])
```
