from setuptools import find_packages, setup


setup(
    name="vcbench",
    version="0.1.0",
    description="VCBench: single-cell perturbation modeling and benchmarking framework",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    package_data={
        "VCBench": ["configs/**/*.yaml", "configs/**/*.yml"],
    },
    python_requires=">=3.10",
)
