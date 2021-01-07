# Natural Reverb - LV2 Plugin

## Installation

You will need: `lvtk-2`, `zita-convolver` (v4), lv2-toolkit, Eigen, and
Boost.Test. Some of these are usually available with your package manager. With
a little hacking of `CMakeLists.txt` Boost can likely be eliminated.

```bash
cmake .
make -j
sudo make install
```

## What does it mean?

For a full description of the math, see
https://michaelganger.org/articles/natural-reverb. A quick summary is that this
is a convolution reverb plugin that algorithmically generates the impulse
response from parameters.
