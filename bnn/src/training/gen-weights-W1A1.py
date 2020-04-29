import os
import finnthesizer as fth

import os
import sys
import finnthesizer as fth

if __name__ == "__main__":
    bnnRoot = "."
    npzFile = bnnRoot + "/train_validation-1w-1a.npz"
    targetDirBin = bnnRoot + "/eye_params/lfcW1A1"
    simdCounts = [64, 32, 64, 8]
    peCounts = [32, 64, 32, 16]
    classes = map(lambda x: str(x), range(2))

    fth.convertFCNetwork(npzFile, targetDirBin, simdCounts, peCounts)
    with open(targetDirBin + "/classes.txt", "w") as f:
        f.write("\n".join(classes))
