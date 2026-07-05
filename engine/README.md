# engine/ — first-party engine-track code

First-party C++ that builds against the vendored vanilla Geant4 tree
(`../geant4-v11.4.2/`, read-only). Nothing here modifies Geant4; see the
engine positioning spec
(`../docs/superpowers/specs/2026-07-05-geant4-engine-positioning-design.md`).

## Building the engine (WSL/Linux)

```bash
# one-time: copy the vendored tree to the Linux filesystem and build
rsync -a /mnt/c/.../Latent-Dirac/geant4-v11.4.2/ ~/latent-dirac-engine/src/
cmake -S ~/latent-dirac-engine/src -B ~/latent-dirac-engine/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/latent-dirac-engine/install \
  -DGEANT4_BUILD_MULTITHREADED=ON \
  -DGEANT4_USE_GDML=ON \
  -DGEANT4_USE_SYSTEM_EXPAT=OFF \
  -DGEANT4_INSTALL_DATA=ON
cmake --build ~/latent-dirac-engine/build && cmake --install ~/latent-dirac-engine/build
```

Visualization/UI/analysis drivers stay off (defaults); datasets (~2.2 GB)
download during the build.

## yieldgen

Antiproton production yield-table generator (proton on iridium,
FTFP_BERT). Output follows the CSV contract in the yieldgen demo spec
and is consumed by `latent_dirac.sources.antiproton_table`.

```bash
source ~/latent-dirac-engine/install/bin/geant4.sh   # PATH + libraries
# provenance: the datasets field of the CSV header comes from this var
export YIELDGEN_DATASETS="$(basename -a $(geant4-config --datasets | awk '{print $3}') | paste -sd,)"
cmake -S engine/yieldgen -B ~/latent-dirac-engine/yieldgen-build -G Ninja \
  -DCMAKE_PREFIX_PATH=$HOME/latent-dirac-engine/install
cmake --build ~/latent-dirac-engine/yieldgen-build
~/latent-dirac-engine/yieldgen-build/yieldgen 200000 pbar_yield.csv 26
```
