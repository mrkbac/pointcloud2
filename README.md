# PointCloud2

PointCloud2 lib for non ROS environment.

[![PyPI version](https://img.shields.io/pypi/v/pointcloud2.svg)](https://pypi.python.org/pypi/pointcloud2/)
[![PyPI license](https://img.shields.io/pypi/l/pointcloud2.svg)](https://pypi.python.org/pypi/pointcloud2/)
[![PyPI download month](https://img.shields.io/pypi/dm/pointcloud2.svg)](https://pypi.python.org/pypi/pointcloud2/)

## Usage

```python
from pointcloud2 import read_points

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

reader = make_reader("file.mcap", decoder_factories=[DecoderFactory()])

for msg in tqdm(reader.iter_decoded_messages(
    topics=["/cloud"],
)):
    cloud = read_points(msg.decoded_message)
    # do something with cloud
    print(cloud['x'])
```
