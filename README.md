# segdtw
Segmental dynamic time warping. Code inspired by https://github.com/slaypni/fastdtw.

## Install
```
pip install segdtw==0.1.0
```

## Usage

```python
from segdtw.distance import segdtw
import numpy as np

query = np.random.random((10, 2))
source = np.random.random((100, 2))

segdtw(source, query, radius=3, dist=2)
```
