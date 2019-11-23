import numpy as np
import os, sys
import math
import argparse
import h5py
from tqdm import tqdm 

jet_types = ['pfjets', 'calojets']
feature_types = ['px', 'py', 'pz', 'energy']

if __name__ == "__main__":
    # Convert the events dataset (with zero padding for jets) to flat jets dataset with no zero padding
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(description='Convert the event dataset to jet dataset.')
    parser.add_argument('--input', type=str,
                        default="step3.h5",
                        help='Input file')
    parser.add_argument('--outdir', type=str,
                        default=cwd,
                        help='Output directory. Default is the current working directory.')
    parser.add_argument('--verbose', action='store_true',
                        default=False,
                        help='Verbose printout for debugging.')

    args = parser.parse_args()
    
    filename = args.input
    outpath = args.outdir
    outfile = os.path.join(outpath, os.path.basename(filename)).replace('.h5','_jets.h5')
    
    infile = h5py.File(filename, "r")
    nEvents = len(infile['pfjets_px'][:])
    print("Reading input file {} with {} events".format(filename, nEvents))
    print("Will save output to file {0}".format(outfile))
    
    output_dict = {}
    # Loop through events
    for iev in tqdm(range(int(nEvents))):
        # Get number of pfjets in each event
        njets = infile['npfjets'][iev]

        # Create corresponding numpy arrays
        for jet in jet_types:
            for ft in feature_types:
                key = jet + '_' + ft
                val = infile[key][iev][:njets] # Only take the non-zero ones
                if key not in output_dict:
                    output_dict[key] = val
                else:
                    output_dict[key] = np.concatenate([output_dict[key], val], axis=0)

    # Save the output
    with h5py.File(outfile, "w") as output_file:
        for jet in jet_types:
            for ft in feature_types:
                key = jet + '_' + ft
                output_file.create_dataset(key, data=output_dict[key], dtype=np.float16)
    print("Saved output to {}".format(outfile))
