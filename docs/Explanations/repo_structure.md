# Repository structure

This document explains the structure of the repository and provides an overview of the different modules.

## Overall structure

The repository is organized as shown below. See the explanations below for information the key files and folders.

```
cellier/
├── .copier-answers.yml        # settings used to create the repo with copier
├── .github/                   # GitHub templates and CI workflows
├── .gitignore
├── .pre-commit-config.yaml    # configuration for the linting
├── LICENSE
├── README.md
├── docs/                      # source files for the documentation
├── examples/                  # example scripts and notebooks
├── pyproject.toml             # packaging metadata
├── scripts/                   # misc. scripts and demos
├── src/                       # Cellier source code
├── tests/                     # pytest tests
└── zensical.toml              # documentation configuration
```

## Cellier library

The Cellier library source code is in `src/cellier`. The Cellier library is organized into several modules, each responsible for a different aspect of the viewer functionality. 

```
src/cellier/
├── _legacy/          # source code from the original version of Cellier
├── _state.py         # immutable state snapshots (DimsState, CameraState, etc.)
├── controller.py     # CellierController — coordinates models, events, and render views
├── convenience/      # utilities for constructing Cellier viewers more easily
├── data/             # data store models and data handling components
├── events/           # event bus and typed event catalogue
├── gui/              # GUI widgets (Qt and Anywidget implementations)
├── logging.py        # debug logging infrastructure
├── paint/            # paint controllers for annotating/labeling
├── py.typed          # PEP 561 marker (declares the package ships type hints)
├── render/           # rendering backend and canvas components
├── scene/            # scene models (cameras, canvas layouts, etc.)
├── transform/        # coordinate transform models (affine, etc.)
├── types.py          # shared annotated types for Pydantic models
├── viewer_model.py   # top-level viewer model (DataManager, ViewerModel)
└── visuals/          # visual models (images, meshes, labels, etc.)
```

### Models

The Cellier models are in several modules grouped by their functionality.

- `cellier.data` contains the data store models and data IO components.
- `cellier.scene` contains the scene models such as cameras, canvas, and dims.
- `cellier.transform` contains the coordinate systems and the transforms
  between them, such as affine transforms.
- `cellier.viewer_model` contains the top-level viewer model.
- `cellier.visuals` contains the visual models such as images, meshes, and labels.

### Views

There are two main view modules: `cellier.gui` and `cellier.render`. `cellier.gui` contains the code for creating the graphical user interface widgets. `cellier.render` contains the code for the rendering backend and canvas.

Within `cellier.gui` there are two implementations of the GUI: `cellier.gui.qt` for the Qt implementation and `cellier.gui.anywidget` for the Anywidget implementation. In general, `cellier.gui.qt` is used for standalone applications while `cellier.gui.anywidget` is used for notebooks such as Jupyter Lab and Marimo. There are some common GUI code in `cellier.gui` such as the protocols and utility functions.

Currently `cellier.render` only contains the `pygfx` implementation. In the future there may be other implementations.

### Controller

The `cellier.controller` module contains the `CellierController` class. The `CellierController` class coordinates the models, events, and GUI and render views.

### Convenience API

The `cellier.convenience` module infrastructure for declarative construction of Cellier viewers. The intention is to create a high-level API for constructing Cellier viewers that is easy to understand and minimizes boilerplate code. Many users will primarily interact with Cellier via the convenience API. However, creating custom viewers by manually composing components will always be a first class path.

### State snapshots

Cellier uses immutable state snapshots to pass the current state of the viewer between different components. For example, if a visual appearance model is updated, the new state snapshot is passed to the GUI view to update the widget. The state snapshots are defined in `cellier._state`.

### Events

Events are used to synchronize the state between the model and views. The events infrastructure is in `cellier.events`.

### Utilities

- `cellier.logging` module contains the debug logging infrastructure. See the [logging explanation](./logging.md) for more information.
- `cellier.paint` module contains the paint controllers for labeling image data.
- `cellier.types` module contains shared annotated types for Pydantic models.

### Legacy

There is a `cellier._legacy` module that contains the source code from the original version of Cellier. This module is kept for reference and backward compatibility, but it is not actively maintained or developed.

## Examples and scripts

The maintained examples demonstrating how to use the library are in `examples`. The `scripts` folder contains miscellaneous scripts and demos that have been used to develop the library, but are not maintained. As the library stabilizes, the useful scripts will be moved to `examples` and the rest will be removed.

## Documentation
The documentation is built with `zensical`. The documentation source files are in `docs` and the configuration is in `zensical.toml`. The Github actions workflow to build and deploy the documentation is in `.github/workflows/docs.yml`.

## Tests

The tests are in `tests` and are organized to match the layout of the `src/cellier` directory. Currently, a lot of the tests are in `tests/v2` as we are in the process of migrating to a new version of the library, but this is a temporary state. All tests are written to be used with pytest.

## Packaging

The packaging metadata are in `pyproject.toml`. The packaging is published to PyPI via the Github actions workflow in `.github/workflows/ci.yml`.