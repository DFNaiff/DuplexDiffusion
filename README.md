# Duplex Diffusion
A model and simulator for 1d component diffusion in biphasic materials (one matrix and one disperse phase),
considering only their diffusion coefficients and their solubilities,
volumetric fraction and mean volume. In progress, but one can install already if he wants.

Depends on boost::math and eigen3. JSON parsing done via "JSON for Modern C++" (header file included here).

# Usage
After installing, just use the simple command line application:

```
$ ./duplexdiffusion tfinal savefile paramfile
```

With tfinal being a number (final time for simulator),
savefile being where the result is stored,
and paramfile being the JSON parameter file.
\
An commented example of a paramfile can be found in params/params.json

# Theory
The theory can be found in tex/document.pdf