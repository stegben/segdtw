# segdtw
Segmental dynamic time warping

## Usage

```python
from segdtw.distance import segdtw
import numpy as np

query = np.random.random((10, 2))
source = np.random.random((100, 2))

segdtw(source, query, radius=3, dist=2)
```
