from setuptools import setup


def readme():
    with open("README.rst") as f:
        return f.read()


setup(
    name="trecs",
    version="0.2.1",
    description="Framework for simulating sociotechnical systems.",
    url="https://github.com/elucherini/t-recs",
    license="MIT",
    author="Eli Lucherini",
    author_email="elucherini@cs.princeton.edu",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=readme(),
    packages=[
        "trecs",
        "trecs.models",
        "trecs.metrics",
        "trecs.components",
        "trecs.base",
        "trecs.random",
        "yt_sim",
        "yt_sim.agents",
    ],
    install_requires=[
        "numpy>=1.17.0",
        "scipy>=1.4.1",
        "networkx>=2.4",
        "tqdm>=4.46.0",
        # pandas>=1.0.5 (loosened from the historical ==1.0.5 pin so the library
        # installs on modern stacks; the code paths used here work on pandas 2.x).
        "pandas>=1.0.5",
        # matplotlib is imported directly by trecs.metrics.measurement.
        "matplotlib>=3.0",
    ],
    extras_require={
        # ImplicitMF depends on lenskit's older `lenskit.algorithms` API. Kept
        # optional so the core library installs cleanly without it.
        "mf": ["lenskit>=0.11.1"],
        # yt_sim's Gymnasium environment (the RL-facing extension).
        "rl": ["gymnasium>=0.29"],
    },
    zip_safe=False,
)
