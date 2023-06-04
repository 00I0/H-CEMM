# H-CEMM
## Diffusion Analysis Project

This project focuses on analyzing diffusion data obtained from experiments on fish. It provides a Python package that includes classes for loading, processing, and visualizing the diffusion data.

### Installation

You can use this project form [this notebook](https://colab.research.google.com/drive/1hF2iO7PVChLhxLPhayTbrtgaHG04MJVo?usp=sharing)

### Usage

#### DiffusionArray Class

The `DiffusionArray` class is a wrapper around a numpy array representing diffusion data. It provides methods for loading data from files, extracting frames or channels.

```python
from diffusion_array import DiffusionArray

# Create a DiffusionArray object from a file
diff_array = DiffusionArray('data.npy')

# Extract a single frame
frame_data = diff_array.frame(0)

# Extract a single channel
channel_data = diff_array.channel(1)

```

#### Reader and Writer classes

The `Reader` class provides a way to instantiate the correct reader based on the file extension. Similarly, the `Writer` class allows you to instantiate the correct writer based on the file extension. This enables easy loading and saving of diffusion data in different formats.

```python
from reader import Reader
from writer import Writer

# Create a reader based on the file extension
reader = Reader.of_type('data.nd2')

# Load the diffusion data using the reader
diff_array = reader.read('data.nd2')

# Create a writer based on the file extension
writer = Writer.of_type('data.npz')

# Save the diffusion data using the writer
writer.save(diff_array, 'output.npz')
```

