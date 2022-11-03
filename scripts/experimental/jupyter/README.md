<!-- Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved. -->

# Jupyter Workflows

This bazel package defines a `:jupyter` target that starts a `jupyter_server` within a bazel sandbox while making all dependencies available. This enables connecting a notebook as a client to run within this server environment.

## Howto run a python notebook within vscode

1. Start the jupyter server with `bazel run //scripts/experimental/jupyter:jupyter`
   (make sure to define missing `deps` for this target if necessary). Record the reported URL with the
   associated token, which have the form

   `http://localhost:8890/?token=c977396227f2d07b923dd53fb1a0823c38df37104ea56264`

2. Load a `.ipynb` file into vscode and use the "Connect to Another Jupyter Server", pointint to the
   reported URL of the server. This also supports adding breakpoint and debugging python-code (including internal DSAI code).
