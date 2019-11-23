# ParticleFlow Regression

Run `python ntuplizer.py` in an CMSSW environment to convert the EDM ROOT file to h5py. I use `10_2_13` (random choice).

The h5py file is structured as follow:
- List of PF jets, sorted by descending pT as in the original EDM file
- List of calo jets closest to the given PF jets above. Note that since there might be fewer calo jets than PF jets in the EDM ROOT file and there is 1 to 1 correspondance between PF jets and Calo jets in the output h5 file, there might be duplication of calo jets in the output.
- List of pixel tracks within the 0.4 radius of the given calo jet. Maximum number of tracks for each jet is set to be 20, however, in most cases, there are between 0 and 3 tracks.
- Max number of PF jets is set to be 200 for each event. For each calo jet, max number of tracks in each jet is 20. Zero padding is applied.

The h5py output is structured per event, with zero padding on jets. To convert this dataset to flat jet structure with no zero padding, run `python events_to_jets_ds.py`. The output of this contains just the PF jets and Calo jets. 

The training is done with `python autoencoder.py`. The saved model can be loaded and tested in `Visualization` notebook.
