# CDTools

CDTools is the internal ptychographic and CDI reconstruction codebase used in the Comin photon scattering group at MIT.

Due to uncertainty about the current patent situation regarding Ptychography, we have refrained from posting the code publicly. We are happy to provide the code upon request. Requests should be directed to [Abe Levitan](alevitan@mit.edu).

## Philosophy

The central design goal of CDTools is to make it easy to blend the more flexible automatic differentiation based reconstruction algorithms with more performant traditional declarative algorithms. Because of this, the central component of CDTools is a set of tools which implement common operations in CDI algorithms using the pytorch package. This allows algorithms based on these tools to be run performantly in a CPU or GPU environment, but more importantly it makes all these tools compatible with pytorch's autograd framework.


Additionally, the code is designed to work close to natively with .cxi files, the current standard for exchange of ptychography and CDI data. This makes it straightforward to load datasets for reconstruction directly from .cxi files, and to save simulated datasets to .cxi files for sharing.


Finally, CDTools defines a standard CDI reconstruction module, which can be quickly fleshed out with premade tools defining propagators, measurements, sample-probe interactions, etc. Reconstructions can be run based only on a defined forward model using automatic differentiation, with very little required parameter tuning. However, specific, declarative reconstruction algorithms can also be defined, allowing reconstructions to proceed by either route or via a combination of automatic differentiation steps and specifically defined algorithms.


## Dependencies


* Numpy
* Scipy
* Matplotlib
* Pytorch
* h5py
* Dateutil


## Installation

CDTools can be installed via pip. Because it is still under active development, it is recommended to install it in developer mode. In the top level folder, run:

```bash
$ pip install -e .
```

## Examples

The following example demonstrates the simplest possible reconstruction, using as an example the "Gold Balls" dataset available at the [CXI database](http://cxidb.org/id-65.html).

```python

import CDTools
from CDTools.tools.plotting import *
import h5py
from matplotlib import pyplot as plt

# Load the dataset
with h5py.File('AuBalls_700ms_30nmStep_3_3SS_filter.cxi','r') as f:
    dataset = CDTools.datasets.Ptycho_2D_Dataset.from_cxi(f)

# Create a "SimplePtycho" model initialized from the dataset
model = CDTools.models.SimplePtycho.from_dataset(dataset)

# Send the model and dataset to the GPU
model.to(device='cuda')
dataset.get_as(device='cuda')

# Run an automatic differentiation reconstruction for 20 iterations
# with default batch size and learning rate
for loss in model.Adam_optimize(20, dataset):
    print(loss)

# And plot the results
plot_amplitude(model.probe)
plot_phase(model.probe)
plot_amplitude(model.obj)
plot_phase(model.obj)
plt.show()
```

The SimplePtycho model exists mostly as a soft introduction to this general pattern. Because it essentially mimics the model for standard ePIE, it is very susceptible to most sources of error that appear in real experiments. We can improve the reconstruction quality dramatically by including a detector background, a few incoherent modes, and reconstructing the true probe positions. All of this is built into the FancyPtycho model. Our improved code looks remarkably similar:

```python
import CDTools
from CDTools.tools.plotting import *
import h5py

# Load the dataset
with h5py.File('AuBalls_700ms_30nmStep_3_3SS_filter.cxi','r') as f:
    dataset = CDTools.datasets.Ptycho_2D_Dataset.from_cxi(f)

# Create a "FancyPtycho" model initialized from the dataset
# Give the model 3 incoherent probe modes, the default is 1
model = CDTools.models.FancyPtycho.from_dataset(dataset,n_modes=3)

# Send the model and dataset to the GPU
model.to(device='cuda')
dataset.get_as(device='cuda')

# Now we run 3 automatic differentiation reconstructions in series,
# lowering the learning rate each time. Here we explicitly set the
# batch size to a more natural setting for this dataset


for i, loss in enumerate(model.Adam_optimize(20, dataset, batch_size=100)):
    print(i,loss)

for i, loss in enumerate(model.Adam_optimize(20, dataset, batch_size=100, lr=0.001)):
    print(i,loss)

for i, loss in enumerate(model.Adam_optimize(30, dataset, batch_size=100, lr=0.0001)):
    print(i,loss)


plot_amplitude(model.probe[0], basis=model.probe_basis.cpu()*1e6)
plot_phase(model.probe[0], basis=model.probe_basis.cpu()*1e6)
plot_amplitude(model.obj, basis=model.probe_basis.cpu()*1e6)
plot_phase(model.obj, basis=model.probe_basis.cpu()*1e6)
plot_translations_from_model(model, dataset)
plt.show()
```

These are two of the default ptychography modules that cover standard situations. However, the entire point of the automatic differentiation approach is that it is straightforward to define new models. Here we demonstrate the definition of a bare-bones module, akin to SimplePtycho, but including a background.

```python

class BackgroundPtycho(CDIModel):

    def __init__(self, ...):
    	# Initialization code to store variables


    def from_dataset(self, dataset):
        # Initialization code to generate a sensible initialization
	# from a specific dataset
	
    
    # This is required to return a stack of exit waves from a stack of
    # pattern indices and translations (for a ptychography model).
    def interaction(self, index, translations):
    	# We first convert the translation to pixel-space
        pix_trans = tools.interactions.translations_to_pixel(self.probe_basis,
                                                             translations)
        pix_trans -= self.min_translation
	# And then use one of the predefined tools to calculate the
	# exit waves
        return tools.interactions.ptycho_2D_round(self.probe_norm * self.probe,
                                                  self.obj,
                                                  pix_trans)


    # Here we only have to choose a propagator - near field or far field
    def forward_propagator(self, wavefields):
        return tools.propagators.far_field(wavefields)


    # Here we define the measurement. This is the only difference from
    # SimplePtycho! We simply pull a quadratic background model off the shelf
    # to use for the background model
    def measurement(self, wavefields):
        return tools.measurements.quadratic_background(wavefields,
                            self.background,
                            detector_slice=self.detector_slice)
    

    # Here we choose what loss function to use for our reconstruction
    def loss(self, sim_data, real_data, mask=None):
        return tools.losses.amplitude_mse(real_data, sim_data,mask=mask)


    def to(self, *args, **kwargs):
    	# Boilerplate code to move model's parameters between datatypes
	# and devices
    
```