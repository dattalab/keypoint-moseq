#!/bin/bash

set -e

# Install and Enable widget extensions configurator
jupyter nbextension install --py jupyter_nbextensions_configurator --sys-prefix
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter nbextension enable --py --sys-prefix  --py qgrid

# Install bokeh extensions
jupyter nbextension install --sys-prefix --symlink --py jupyter_bokeh
jupyter nbextension enable jupyter_bokeh --py --sys-prefix
