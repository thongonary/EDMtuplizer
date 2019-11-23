import ROOT
import sys, os
import numpy as np
from DataFormats.FWLite import Events, Handle
import math
import argparse
import h5py
from tqdm import tqdm

class HandleLabel:
    def __init__(self, dtype, label):
        self.handle = Handle(dtype)
        if isinstance(label, tuple) or isinstance(label, list):
            self.label = tuple(label)
        else:
            self.label = (label, )

    def getByLabel(self, event):
        event.getByLabel(self.label, self.handle)

    def product(self):
        return self.handle.product()

class EventDesc:
    def __init__(self):
        # From edmDumpEventContent
        self.caloTower = HandleLabel("BXVector<l1t::CaloTower>", ("caloStage2Digis", "CaloTower"))
        self.caloJet = HandleLabel("std::vector<reco::CaloJet>", "ak4CaloJets")
        self.pfJet = HandleLabel("std::vector<reco::PFJet>", "ak4PFJetsCHS") # assuming we want CHS
        self.pixelTrack = HandleLabel("std::vector<reco::Track>", "pixelTracks")

    def get(self, event):
        self.caloTower.getByLabel(event)
        self.caloJet.getByLabel(event)
        self.pfJet.getByLabel(event)
        self.pixelTrack.getByLabel(event)

class Output:
    def __init__(self, outfile, maxEvents):
        self.maxEvents = maxEvents

        # PF Jets
        self.npfjets = np.zeros(self.maxEvents, dtype=np.uint32)
        self.maxpfjets = 200
        self.pfjets_pt = np.zeros((self.maxEvents, self.maxpfjets), dtype=np.float32)
        self.pfjets_eta = np.zeros((self.maxEvents, self.maxpfjets), dtype=np.float32)
        self.pfjets_phi = np.zeros((self.maxEvents, self.maxpfjets), dtype=np.float32)
        self.pfjets_energy = np.zeros((self.maxEvents, self.maxpfjets), dtype=np.float32)
        self.pfjets_px = np.zeros((self.maxEvents, self.maxpfjets), dtype=np.float32)
        self.pfjets_py = np.zeros((self.maxEvents, self.maxpfjets), dtype=np.float32)
        self.pfjets_pz = np.zeros((self.maxEvents, self.maxpfjets), dtype=np.float32)

        # Calo Jets
        self.maxcalojets = 200
        self.calojets_pt = np.zeros((self.maxEvents, self.maxcalojets), dtype=np.float32)
        self.calojets_eta = np.zeros((self.maxEvents, self.maxcalojets), dtype=np.float32)
        self.calojets_phi = np.zeros((self.maxEvents, self.maxcalojets), dtype=np.float32)
        self.calojets_energy = np.zeros((self.maxEvents, self.maxcalojets), dtype=np.float32)
        self.calojets_px = np.zeros((self.maxEvents, self.maxcalojets), dtype=np.float32)
        self.calojets_py = np.zeros((self.maxEvents, self.maxcalojets), dtype=np.float32)
        self.calojets_pz = np.zeros((self.maxEvents, self.maxcalojets), dtype=np.float32)

        # Pixel tracks inside the calo jet cone
        self.maxtracks = 20
        self.tracks_pt = np.zeros((self.maxEvents, self.maxcalojets, self.maxtracks), dtype=np.float32)
        self.tracks_eta = np.zeros((self.maxEvents, self.maxcalojets, self.maxtracks), dtype=np.float32)
        self.tracks_phi = np.zeros((self.maxEvents, self.maxcalojets, self.maxtracks), dtype=np.float32)
        self.tracks_px = np.zeros((self.maxEvents, self.maxcalojets, self.maxtracks), dtype=np.float32)
        self.tracks_py = np.zeros((self.maxEvents, self.maxcalojets, self.maxtracks), dtype=np.float32)
        self.tracks_pz = np.zeros((self.maxEvents, self.maxcalojets, self.maxtracks), dtype=np.float32)
        
        # Calo Tower
        # To be updated

    def save(self, filename):
        with h5py.File(filename, "w") as outfile:
            outfile.create_dataset("npfjets", data=self.npfjets, dtype=np.int32)
            outfile.create_dataset("pfjets_pt", data=self.pfjets_pt, dtype=np.float16)
            outfile.create_dataset("pfjets_eta", data=self.pfjets_eta, dtype=np.float16)
            outfile.create_dataset("pfjets_phi", data=self.pfjets_phi, dtype=np.float16)
            outfile.create_dataset("pfjets_energy", data=self.pfjets_energy, dtype=np.float16)
            outfile.create_dataset("pfjets_px", data=self.pfjets_px, dtype=np.float16)
            outfile.create_dataset("pfjets_py", data=self.pfjets_py, dtype=np.float16)
            outfile.create_dataset("pfjets_pz", data=self.pfjets_pz, dtype=np.float16)

            outfile.create_dataset("calojets_pt", data=self.calojets_pt, dtype=np.float16)
            outfile.create_dataset("calojets_eta", data=self.calojets_eta, dtype=np.float16)
            outfile.create_dataset("calojets_phi", data=self.calojets_phi, dtype=np.float16)
            outfile.create_dataset("calojets_energy", data=self.calojets_energy, dtype=np.float16)
            outfile.create_dataset("calojets_px", data=self.calojets_px, dtype=np.float16)
            outfile.create_dataset("calojets_py", data=self.calojets_py, dtype=np.float16)
            outfile.create_dataset("calojets_pz", data=self.calojets_pz, dtype=np.float16)

            outfile.create_dataset("tracks_pt", data=self.tracks_pt, dtype=np.float16)
            outfile.create_dataset("tracks_eta", data=self.tracks_eta, dtype=np.float16)
            outfile.create_dataset("tracks_phi", data=self.tracks_phi, dtype=np.float16)
            outfile.create_dataset("tracks_px", data=self.tracks_px, dtype=np.float16)
            outfile.create_dataset("tracks_py", data=self.tracks_py, dtype=np.float16)
            outfile.create_dataset("tracks_pz", data=self.tracks_pz, dtype=np.float16)

        print("Saved output to {}".format(filename))

    def clear(self):
        self.tracks_pt.fill(0)
        self.tracks_phi.fill(0)
        self.tracks_eta.fill(0)
        self.tracks_px.fill(0)
        self.tracks_py.fill(0)
        self.tracks_pz.fill(0)

        self.npfjets.fill(0)
        self.pfjets_pt.fill(0)
        self.pfjets_eta.fill(0)
        self.pfjets_phi.fill(0)
        self.pfjets_energy.fill(0)
        self.pfjets_px.fill(0)
        self.pfjets_py.fill(0)
        self.pfjets_pz.fill(0)

        self.calojets_pt.fill(0)
        self.calojets_eta.fill(0)
        self.calojets_phi.fill(0)
        self.calojets_energy.fill(0)
        self.calojets_px.fill(0)
        self.calojets_py.fill(0)
        self.calojets_pz.fill(0)

# define deltaR
def validatePhi(x):
    while x >= math.pi:
        x -= 2.*math.pi
    while x < -math.pi:
        x += 2.*math.pi
    return x

def deltaR(a,b):
    deta = a.eta()-b.eta()
    dphi = validatePhi(a.phi()-b.phi());
    return math.sqrt(deta*deta + dphi*dphi)

if __name__ == "__main__":
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(description='Ntuplizing EDM file.')
    parser.add_argument('--input', type=str,
                        default="/afs/cern.ch/work/a/adiflori/public/forMaurizio/step3.root",
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
    outfile = os.path.join(outpath, os.path.basename(filename)).replace('.root','.h5')
    events = Events(filename)
    nEvents = events.size()
    print("Reading input file {} with {} events".format(filename, nEvents))
    print("Will save output to file {0}".format(outfile))

    evdesc = EventDesc()
    output = Output(outfile, nEvents)

    # loop over events
    with tqdm(total=nEvents) as pbar:
        for iev, event in enumerate(events):
            eid = event.object().id()
            eventId = (eid.run(), eid.luminosityBlock(), int(eid.event()))
            evdesc.get(event)

            # Get the list of pf jets
            output.npfjets[iev] = len(evdesc.pfJet.product())
            for i, pfJ in enumerate(evdesc.pfJet.product()):
                if i > output.maxpfjets:
                    print("More than {} jets, move on to the next event".format(output.maxpfjets))
                    break

                # Fill the output PF Jets, already sorted by descending pT
                output.pfjets_pt[iev, i] = pfJ.pt()
                output.pfjets_eta[iev, i] = pfJ.eta()
                output.pfjets_phi[iev, i] = pfJ.phi()
                output.pfjets_energy[iev, i] = pfJ.energy()
                output.pfjets_px[iev, i] = pfJ.px()
                output.pfjets_py[iev, i] = pfJ.py()
                output.pfjets_pz[iev, i] = pfJ.pz()

                # Find the closest calo jet to the given pf jet
                if args.verbose:
                    print("PF Jet #{}: pt {} eta {} phi {}".format(i, pfJ.pt(), pfJ.eta(), pfJ.phi()))
                minDR = 1e8
                bestJ = 0
                for j, caloJ in enumerate(evdesc.caloJet.product()):
                    dR = deltaR(caloJ, pfJ)
                    if dR < minDR:
                        minDR = dR
                        bestJ = j
                closestCJ = evdesc.caloJet.product().at(bestJ)
                if args.verbose:
                    print("\tMatching caloJet #{}: pt {} eta {} phi {}. Distance = {}".format(bestJ, closestCJ.pt(),
                                                                                closestCJ.eta(),
                                                                                closestCJ.phi(),
                                                                                minDR))
                output.calojets_pt[iev, i] = closestCJ.pt()
                output.calojets_eta[iev, i] = closestCJ.eta()
                output.calojets_phi[iev, i] = closestCJ.phi()
                output.calojets_energy[iev, i] = closestCJ.energy()
                output.calojets_px[iev, i] = closestCJ.px()
                output.calojets_py[iev, i] = closestCJ.py()
                output.calojets_pz[iev, i] = closestCJ.pz()

                # Find all the pixel tracks of the given calo jet cone
                itrack = 0
                for t, ptrack in enumerate(evdesc.pixelTrack.product()):
                    if itrack > output.maxtracks:
                        print("More than {} tracks, move on to the next jet".format(output.maxtracks))
                        break

                    dR = deltaR(closestCJ, ptrack)
                    if dR < 0.4:
                        if args.verbose:
                            print("\t\tContaining track #{}: pt {} eta {} phi {}. dR = {}".format(t, ptrack.pt(),
                                                                                ptrack.eta(),
                                                                                ptrack.phi(), dR))
                        output.tracks_pt[iev, i, itrack] = ptrack.pt()
                        output.tracks_eta[iev, i, itrack] = ptrack.eta()
                        output.tracks_phi[iev, i, itrack] = ptrack.phi()
                        output.tracks_px[iev, i, itrack] = ptrack.px()
                        output.tracks_py[iev, i, itrack] = ptrack.py()
                        output.tracks_pz[iev, i, itrack] = ptrack.pz()
                        itrack += 1

            pbar.update(1)
    
    # Save the output
    output.save(outfile)
