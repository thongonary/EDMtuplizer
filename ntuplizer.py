import ROOT
import sys, os
import numpy as np
from DataFormats.FWLite import Events, Handle

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
    def __init__(self, outfile):
        self.tfile = ROOT.TFile(outfile, "RECREATE")
        self.tree = ROOT.TTree("tree", "tree")
       
        # Pixel Tracks
        self.ntracks = np.zeros(1, dtype=np.uint32)
        self.maxtracks = 5000
        self.tracks_phi = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_eta = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_dxy = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_dsz = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_inner_eta = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_inner_phi = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_outer_eta = np.zeros(self.maxtracks, dtype=np.float32)
        self.tracks_outer_phi = np.zeros(self.maxtracks, dtype=np.float32)
        
        self.tree.Branch("ntracks", self.ntracks, "ntracks/i")
        self.tree.Branch("tracks_phi", self.tracks_phi, "tracks_phi[ntracks]/F")
        self.tree.Branch("tracks_eta", self.tracks_eta, "tracks_eta[ntracks]/F")
        self.tree.Branch("tracks_dxy", self.tracks_dxy, "tracks_dxy[ntracks]/F")
        self.tree.Branch("tracks_dsz", self.tracks_dsz, "tracks_dsz[ntracks]/F")
        self.tree.Branch("tracks_outer_eta", self.tracks_outer_eta, "tracks_outer_eta[ntracks]/F")
        self.tree.Branch("tracks_outer_phi", self.tracks_outer_phi, "tracks_outer_phi[ntracks]/F")
        self.tree.Branch("tracks_inner_eta", self.tracks_inner_eta, "tracks_inner_eta[ntracks]/F")
        self.tree.Branch("tracks_inner_phi", self.tracks_inner_phi, "tracks_inner_phi[ntracks]/F")
       
        # PF Jets
        self.npfjets = np.zeros(1, dtype=np.uint32)
        self.maxpfjets = 200
        self.pfjets_pt = np.zeros(self.maxpfjets, dtype=np.float32)
        self.pfjets_eta = np.zeros(self.maxpfjets, dtype=np.float32)
        self.pfjets_phi = np.zeros(self.maxpfjets, dtype=np.float32)
        self.pfjets_energy = np.zeros(self.maxpfjets, dtype=np.float32)
        
        self.tree.Branch("npfjets", self.npfjets, "npfjets/i")
        self.tree.Branch("pfjets_pt", self.pfjets_pt, "pfjets_pt[npfjets]/F")
        self.tree.Branch("pfjets_eta", self.pfjets_eta, "pfjets_eta[npfjets]/F")
        self.tree.Branch("pfjets_phi", self.pfjets_phi, "pfjets_phi[npfjets]/F")
        self.tree.Branch("pfjets_energy", self.pfjets_energy, "pfjets_energy[npfjets]/F")
       
        # Calo Jets
        self.ncalojets = np.zeros(1, dtype=np.uint32)
        self.maxcalojets = 200
        self.calojets_pt = np.zeros(self.maxcalojets, dtype=np.float32)
        self.calojets_eta = np.zeros(self.maxcalojets, dtype=np.float32)
        self.calojets_phi = np.zeros(self.maxcalojets, dtype=np.float32)
        self.calojets_energy = np.zeros(self.maxcalojets, dtype=np.float32)
        
        self.tree.Branch("ncalojets", self.ncalojets, "ncalojets/i")
        self.tree.Branch("calojets_pt", self.calojets_pt, "calojets_pt[ncalojets]/F")
        self.tree.Branch("calojets_eta", self.calojets_eta, "calojets_eta[ncalojets]/F")
        self.tree.Branch("calojets_phi", self.calojets_phi, "calojets_phi[ncalojets]/F")
        self.tree.Branch("calojets_energy", self.calojets_energy, "calojets_energy[ncalojets]/F")
      
        # Calo Tower
        # To be updated

    def close(self):
        self.tfile.Write()
        self.tfile.Close()

    def clear(self):
        
        self.ntracks[0] = 0
        self.tracks_phi[:] = 0
        self.tracks_eta[:] = 0
        self.tracks_dxy[:] = 0
        self.tracks_dsz[:] = 0
        self.tracks_outer_eta[:] = 0
        self.tracks_outer_phi[:] = 0
        self.tracks_inner_eta[:] = 0
        self.tracks_inner_phi[:] = 0
        
        self.npfjets[0] = 0
        self.pfjets_pt[:] = 0
        self.pfjets_eta[:] = 0
        self.pfjets_phi[:] = 0
        self.pfjets_energy[:] = 0
        
        self.ncalojets[0] = 0
        self.calojets_pt[:] = 0
        self.calojets_eta[:] = 0
        self.calojets_phi[:] = 0
        self.calojets_energy[:] = 0

# define deltaR
from math import hypot, pi
def deltaR(a,b):
    dphi = abs(a.phi()-b.phi());
    if dphi < pi: dphi = 2*pi-dphi
    return hypot(a.eta()-b.eta(),dphi)

if __name__ == "__main__":

    filename = "/afs/cern.ch/work/a/adiflori/public/forMaurizio/step3.root"
    outpath = "~/."
    outfile = os.path.join(outpath, os.path.basename(filename))
    events = Events(filename)
    print("Reading input file {0}".format(filename))
    print("Will save output to file {0}".format(outfile)) 

    evdesc = EventDesc()
    output = Output(outfile)

    num_events = events.size()
    
    # loop over events
    for iev, event in enumerate(events):
        #if iev > 10:
        #    break
        eid = event.object().id()
        if iev%10 == 0:
            print("Event {0}/{1}".format(iev, num_events))
        eventId = (eid.run(), eid.luminosityBlock(), int(eid.event()))
         
        evdesc.get(event)
        
        # For every pf jet, look for the nearest calo jet
        for i, pfJ in enumerate(evdesc.pfJet.product()):
            print("PF Jet #{}: pt {} eta {} phi {}".format(i, pfJ.pt(), pfJ.eta(), pfJ.phi()))
            minDR = 1e8
            bestJ = 0
            for j, caloJ in enumerate(evdesc.caloJet.product()):
                dR = deltaR(caloJ, pfJ)
                if dR < minDR:
                    minDR = dR
                    bestJ = j
            bestmatch = evdesc.caloJet.product().at(bestJ)
            print("\tMatching caloJet #{}: pt {} eta {} phi {}. Distance = {}".format(bestJ, bestmatch.pt(), 
                                                                    bestmatch.eta(),
                                                                    bestmatch.phi(),
                                                                    minDR))

            # Find all the pixel tracks of the given calo jet cone
            for t, ptrack in enumerate(evdesc.pixelTrack.product()):
                dR = deltaR(bestmatch, ptrack)
                print("dR = {}".format(dR))
                if dR < 0.4:
                    print("\tTrack #{}: pt {} eta {} phi {}".format(t, ptrack.pt(),
                                                                    ptrack.eta(),
                                                                    ptrack.phi()))


         # caloTower is empty on the first event, not sure why
#        for i, p in enumerate(evdesc.caloTower.product()):
#            print("First BX {}, last BX {}".format(p.getFirstBX(), p.getLastBX()))
#            for bx in xrange(p.getFirstBX(), p.getLastBX()+1):
#                for j in xrange(p.size(bx)):
#                    ctobject = p.at(bx, j)
#                    print("Calo Tower # {}: BX {} - {}, pt {}, eta {}, phi".format(i, bx, j, ctobject.pt(), ctobject.eta(), ctobject.phi()))
        
        break

        output.clear()
        output.tree.Fill()
    pass 
    #end of event loop 

    output.close()
