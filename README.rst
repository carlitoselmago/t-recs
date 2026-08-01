T-RECS (Tool for RecSys Simulation)
=====================================

.. image:: https://i.imgur.com/3ZRDVaD.png
  :width: 400
  :alt: Picture of T-Rex next to letters T-RECS

A library for using agent-based modeling to perform simulation studies of sociotechnical systems.

Installation
------------

System requirements
###################

Currently, the simulator has only been tested extensively on MacOS 10.15 and Ubuntu 20.04.
This simulator supports Python 3.7+ and it has not been tested with older versions of Python 3. If you have not configured a Python environment locally, please follow Scipy's `instructions for installing a scientific Python distribution`_.

.. _instructions for installing a scientific Python distribution: https://scipy.org/install.html

If you do not have Python 3.7+ installed, you can create a new environment with Python 3.7 by running the following command in terminal:

..  code-block:: bash

    conda create --name py37 python=3.7

To ensure the example Jupyter notebooks run in your Python 3.7 environment, follow `the instructions from this blog post`_. **Note**: you will also need ``pandas`` to run the example notebooks. As of December 2020, we recommend installing ``pandas v1.0.5`` using the command: ``pip install 'pandas==1.0.5'``. This will help avoid package conflicts between ``pandas`` and ``pylint`` if you also plan on contributing to ``trecs`` and running tests.

.. _the instructions from this blog post: https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084

For users
#########

To install the simulator, you will need the Python package manager, ``pip``. After activating your virtual environment, run the following command in a terminal:

..  code-block:: bash

  pip install trecs

For developers
##############

If you'd like to install the latest version of ``trecs`` based on what is currently in the main branch of the repository, run the following commands after activating your virtual environment:

..  code-block:: bash

  git clone https://github.com/elucherini/t-recs.git
  cd t-recs
  pip install -e .

Additionally, you may run ``pip install -r requirements-dev.txt`` to install a few additional dependencies that will be useful for linting, testing, etc.

**Optional extras.** Two heavy/optional dependencies are installed on demand:

..  code-block:: bash

  pip install -e .[mf]   # enables the lenskit-based ImplicitMF model
  pip install -e .[rl]   # enables the Gymnasium environment used by yt_sim (see below)

``trecs`` installs and runs without either. If ``lenskit`` is missing or
incompatible (its older ``lenskit.algorithms`` API is unavailable on recent
releases), only :class:`trecs.models.ImplicitMF` becomes unavailable and a
warning is emitted; every other model is unaffected.

Documentation
**************
If you would like to edit the documentation, see the ``docs/`` folder. To build the documentation on your local folder, you will need to install ``sphinx`` and the ``sphinx-rtd-theme`` via ``pip``. Next, ``cd`` into the ``docs`` folder and run ``make html``. The output of the command should tell you where the compiled HTML documentation is located.

.. _sphinx: https://www.sphinx-doc.org/en/master/usage/installation.html
.. _sphinx-rtd-theme: https://pypi.org/project/sphinx-rtd-theme/

Tutorials
----------
Examples of how to use the simulator can be found in the notebooks below:

- `Quick start`_: start here for a brief introduction.
- `Complete guide`_: an overview of the main concepts of the system.
- Advanced guide - `building a model`_: an introduction to adding your own models on top of the system.
- Advanced guide - `adding metrics`_: an example of how to add new metrics to a model.

.. _Quick start: https://github.com/elucherini/t-recs/blob/main/examples/quick-start.ipynb
.. _Complete guide: https://github.com/elucherini/t-recs/blob/main/examples/complete-guide.ipynb
.. _building a model: https://github.com/elucherini/t-recs/blob/main/examples/advanced-models.ipynb
.. _adding metrics: https://github.com/elucherini/t-recs/blob/main/examples/advanced-metrics.ipynb

Please check the examples_ directory for more notebooks.

.. _examples: examples/

Example usage
-------------

..  code-block:: bash

  import trecs

  recsys = trecs.models.ContentFiltering()
  recsys.run(timesteps=10)
  measurements = recsys.get_measurements()

yt_sim: a YouTube-like recommendation environment
-------------------------------------------------

``yt_sim`` is an extension built *on top of* T-RECS (it lives in the top-level
``yt_sim/`` package and subclasses T-RECS components rather than modifying them)
that turns the simulator into a black-box, YouTube-like environment for
reinforcement-learning agents. An external agent acts as the *viewer*: at each
step it sees the currently playing video and a slate of suggested videos, and it
chooses an action. It is **not** told what any action does -- only the *next
slate* reveals the consequences, exactly as a real viewer would experience.

What it adds
############

- **Thumbnail-embedding items** (:class:`yt_sim.EmbeddingItems`): items carry a
  fixed-length embedding (e.g. a CLIP thumbnail encoding) instead of, or
  alongside, T-RECS's default attribute vectors. Supply a precomputed
  ``[n_items, embed_dim]`` matrix, or flip a flag to fall back to the original
  random-attribute path for testing.
- **Two-stage recommendation funnel** (:class:`yt_sim.FunnelRecommender`, a
  ``BaseRecommender`` subclass): Stage 1 does k-NN candidate generation over item
  embeddings; Stage 2 ranks the candidates with a pluggable scoring function
  (similarity + popularity + recency by default). It still runs under the
  standard T-RECS ``run()`` loop.
- **Two-tier action space** with pluggable effects
  (:mod:`yt_sim.action_effects`): current-video actions (``watch_full``,
  ``skip``, ``like`` -- ``like`` gated off by default) and suggested-video
  actions (``click``, ``not_interested``). All action *semantics* live in a
  swappable effect module, never in the step loop or the agent-facing API.
- **"Not interested" neighborhood masking** (:class:`yt_sim.NotInterestedMask`):
  suppresses the exact item permanently and its k nearest neighbors for a
  configurable number of steps (neighbors resurface afterwards, mirroring how
  YouTube's "Not interested" is understood to work in practice).
- **Continuous watch simulation** (:class:`yt_sim.WatchModel`): an internal
  engagement fraction, a probabilistic function of preference-item similarity,
  that scales how far the viewer's preference drifts -- never exposed to the
  agent.
- **Gymnasium environment** (``yt_sim.YouTubeSimEnv``): reward-agnostic
  (``reward`` is always ``0.0``) so your agent computes its own intrinsic reward
  from the returned observations.

Install
#######

..  code-block:: bash

  pip install -e .[rl]   # trecs + gymnasium

Quick start with the random-agent baseline
##########################################

..  code-block:: python

  import numpy as np
  from yt_sim import random_embeddings   # placeholder; use load_embeddings() for real CLIP vectors
  from yt_sim.env import YouTubeSimEnv
  from yt_sim.agents import RandomAgent

  # [n_items, embed_dim] embedding matrix (swap in real thumbnail embeddings)
  embeddings = random_embeddings(n_items=500, embed_dim=32, seed=0)

  env = YouTubeSimEnv(embeddings, slate_size=5, max_steps=200, seed=0)
  agent = RandomAgent(seed=0)

  obs, info = env.reset(seed=0)
  # obs = {"current_embedding": (embed_dim,),
  #        "slate_embeddings": (slate_size, embed_dim),
  #        "action_mask": (n_actions,)}   # 1 = action available; meanings are opaque
  done = False
  while not done:
      action = agent.act(obs)                 # samples a valid action from the mask
      obs, reward, terminated, truncated, info = env.step(action)
      # reward is always 0.0 -- compute your own intrinsic reward from `obs` here
      done = terminated or truncated

To load real embeddings, replace ``random_embeddings(...)`` with
``yt_sim.load_embeddings("thumbnails.npy")`` (a thin ``numpy.load`` stub you can
adapt to your dataset).

A runnable baseline script prints rollout statistics:

..  code-block:: bash

  python examples/random_agent_baseline.py --items 500 --dim 32 --steps 200 --episodes 3

Tests
#####

..  code-block:: bash

  # original T-RECS suite
  cd trecs/tests && pytest
  # yt_sim extension suite
  pytest yt_sim/tests

Documentation
--------------

A first draft of the documentation is available `here`_. In its current version, the documentation can be used as a supplement to exploring details in the code. Currently, the tutorials in examples_ might be a more useful and centralized resource to learn how to use the system.

.. _here: https://elucherini.github.io/t-recs/index.html
.. _examples: examples/


Contributing
--------------

Thanks for your interest in contributing! Check out the guidelines for contributors in `CONTRIBUTING.md`_.

.. _CONTRIBUTING.md: https://github.com/elucherini/t-recs/blob/main/CONTRIBUTING.md
