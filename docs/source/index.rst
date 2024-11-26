====================================
NanoFed - Federated Learning Library
====================================

.. note::
   **Version**: 0.1.0 |
   **License**: GPL-3.0 |
   **Python**: >=3.10

----

.. admonition:: Alpha Release
   :class: caution

   NanoFed is currently in alpha. APIs may change in future versions.

Overview
--------

**NanoFed** is a lightweight, modular federated learning library. Built on PyTorch, it enables distributed model training while preserving data privacy.

.. code-block:: bash
   :caption: Quick Installation
   :emphasize-lines: 1

   pip install nanofed
   poetry install nanofed  # Using poetry

Key Features
------------

.. grid:: 2

   .. grid-item-card:: 🔒 Privacy First
      :class-card: sd-border-0.5 sd-shadow-lg

      - Client data never leaves local devices
      - Secure model update transmission
      - Privacy-preserving aggregation

   .. grid-item-card:: 🚀 Easy to Use
      :class-card: sd-border-0.5 sd-shadow-lg

      - Simple, intuitive API
      - PyTorch integration
      - Clear documentation

   .. grid-item-card:: 🔧 Flexible
      :class-card: sd-border-0.5 sd-shadow-lg

      - Custom model support
      - Pluggable aggregation strategies
      - Extensible architecture

   .. grid-item-card:: 💻 Production Ready
      :class-card: sd-border-0.5 sd-shadow-lg

      - Async communication
      - Robust error handling
      - Comprehensive logging

Quick Example
-------------

Train a model using federated learning in just a few lines of code:

.. code-block:: python
   :caption: Basic Training Setup
   :linenos:

   import asyncio
   from pathlib import Path

   from nanofed import HTTPClient
   from nanofed import TorchTrainer, TrainingConfig

   async def run_client(client_id: str, server_url: str):
      training_config = TrainingConfig(
         epochs=1,
         batch_size=256,
         learning_rate=0.1
      )
      async with HTTPClient(server_url, client_id) as client:
         model_state, _ = await client.fetch_global_model()
         await client.submit_update(model, metrics)


   if __name__ == "__main__":
      asyncio.run(run_client("client1", "http://localhost:8080"))

Components
----------

.. tab-set::

   .. tab-item:: Client Layer
      :sync: tech

      The client component handles local training and server communication:

      - Local model training
      - Secure update submission
      - Data privacy preservation
      - Training metrics collection

   .. tab-item:: Server Layer
      :sync: tech

      The server coordinates the federated process:

      - Global model distribution
      - Update aggregation
      - Training coordination
      - Client synchronization

   .. tab-item:: Coordinator
      :sync: tech

      The coordinator manages the overall training process:

      - Round management
      - Client tracking
      - Progress monitoring
      - Checkpointing

Installation
------------

.. dropdown:: Basic Installation
   :icon: package
   :color: primary
   :animate: fade-in

   Install using pip:

   .. code-block:: bash

      pip install nanofed

.. dropdown:: Development Installation
   :icon: code
   :color: primary
   :animate: fade-in

   For development:

   .. code-block:: bash

      git clone https://github.com/camille-004/nanofed.git
      cd nanofed
      make install

Requirements
------------

Required Dependencies
~~~~~~~~~~~~~~~~~~~~~

- Python >=3.10
- PyTorch
- aiohttp
- pydantic

Getting Help
------------


.. grid:: 3

   .. grid-item-card:: 🐛 **Issue Tracker**
      :link: https://github.com/camille-004/nanofed/issues
      :link-type: url
      :class-card: sd-border-0.5 sd-shadow-sm

      Report bugs and request features on our GitHub issue tracker.

   .. grid-item-card:: 📚 **Documentation**
      :link: https://nanofed.readthedocs.io
      :link-type: url
      :class-card: sd-border-0.5 sd-shadow-sm

      Read our comprehensive documentation and guides.

   .. grid-item-card:: 📖 **Source Code**
      :link: https://github.com/camille-004/nanofed
      :link-type: url
      :class-card: sd-border-0.5 sd-shadow-sm

      Browse the source code on GitHub.

.. warning::

   This is an alpha release. While the core functionality is stable, APIs may change
   in future versions. Please report any issues you encounter.

License
-------

NanoFed is available under the GPL-3.0 License.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Documentation
