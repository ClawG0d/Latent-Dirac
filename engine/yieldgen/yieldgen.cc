// yieldgen: antiproton production yield-table generator.
//
// First-party engine-track application (NOT part of the vendored Geant4
// tree). Fires protons at an iridium cylinder with the FTFP_BERT
// reference physics list and records the phase space of every
// antiproton exiting the target surface into the CSV contract defined in
// docs/superpowers/specs/2026-07-05-engine-yieldgen-demo-design.md.
//
// Usage: yieldgen <n_events> <out.csv> [momentum_gev_c=26]
//
// The target is a parameterized AD-like stand-in (r = 1.5 mm,
// half-length 27.5 mm), not facility engineering. Records are exit
// events: an antiproton that re-enters and exits again appears once per
// exit (rare; acceptable for the table-based fidelity tier).

#include <climits>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4PhysListFactory.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Tubs.hh"
#include "G4UserSteppingAction.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4Version.hh"

namespace {

constexpr double kTargetRadius = 1.5 * mm;
constexpr double kTargetHalfLength = 27.5 * mm;

std::ofstream g_out;
std::mutex g_outMutex;

void RecordExit(double xm, double ym, double zm, double pxg, double pyg, double pzg) {
  std::lock_guard<std::mutex> lock(g_outMutex);
  g_out << xm << ',' << ym << ',' << zm << ',' << pxg << ',' << pyg << ',' << pzg << '\n';
}

std::string DatasetBasenames() {
  // Preferred: the runner precomputes the list, e.g.
  //   YIELDGEN_DATASETS="$(basename -a $(geant4-config --datasets \
  //     | awk '{print $3}') | paste -sd,)"
  // (Geant4 11.x locates datasets from the install prefix, so the
  // classic G4*DATA env vars are usually unset; they are kept as a
  // fallback for custom-data setups.)
  if (const char* preset = std::getenv("YIELDGEN_DATASETS")) return preset;

  const char* vars[] = {"G4LEDATA",          "G4ENSDFSTATEDATA", "G4LEVELGAMMADATA",
                        "G4PARTICLEXSDATA",  "G4SAIDXSDATA",     "G4NEUTRONHPDATA",
                        "G4RADIOACTIVEDATA", "G4INCLDATA",       "G4ABLADATA"};
  std::string joined;
  for (const char* var : vars) {
    const char* value = std::getenv(var);
    if (value == nullptr) continue;
    std::string path(value);
    const auto slash = path.find_last_of('/');
    if (!joined.empty()) joined += ",";
    joined += (slash == std::string::npos) ? path : path.substr(slash + 1);
  }
  return joined.empty() ? "unknown" : joined;
}

class DetectorConstruction final : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    auto* vacuum = nist->FindOrBuildMaterial("G4_Galactic");
    auto* iridium = nist->FindOrBuildMaterial("G4_Ir");

    auto* worldSolid = new G4Box("world", 0.5 * m, 0.5 * m, 0.5 * m);
    auto* worldLV = new G4LogicalVolume(worldSolid, vacuum, "world");
    auto* worldPV = new G4PVPlacement(nullptr, {}, worldLV, "world", nullptr, false, 0);

    auto* targetSolid = new G4Tubs("target", 0.0, kTargetRadius, kTargetHalfLength, 0.0, 360.0 * deg);
    auto* targetLV = new G4LogicalVolume(targetSolid, iridium, "target");
    new G4PVPlacement(nullptr, {}, targetLV, "target", worldLV, false, 0);

    return worldPV;
  }
};

class PrimaryGenerator final : public G4VUserPrimaryGeneratorAction {
 public:
  explicit PrimaryGenerator(double momentumGeV) {
    fGun.SetParticleDefinition(G4ParticleTable::GetParticleTable()->FindParticle("proton"));
    fGun.SetParticleMomentumDirection({0.0, 0.0, 1.0});
    fGun.SetParticleMomentum(momentumGeV * GeV);
    fGun.SetParticlePosition({0.0, 0.0, -kTargetHalfLength - 10.0 * mm});
  }

  void GeneratePrimaries(G4Event* event) override { fGun.GeneratePrimaryVertex(event); }

 private:
  G4ParticleGun fGun;
};

class ExitRecorder final : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    const auto* track = step->GetTrack();
    if (track->GetParticleDefinition()->GetPDGEncoding() != -2212) return;

    const auto* pre = step->GetPreStepPoint();
    const auto* post = step->GetPostStepPoint();
    const auto* postVolume = post->GetPhysicalVolume();
    if (postVolume == nullptr) return;  // leaving the world

    if (fTarget == nullptr) {
      fTarget = G4LogicalVolumeStore::GetInstance()->GetVolume("target");
    }
    const bool leavingTarget = pre->GetPhysicalVolume()->GetLogicalVolume() == fTarget &&
                               postVolume->GetLogicalVolume() != fTarget;
    if (!leavingTarget) return;

    const auto position = post->GetPosition();
    const auto momentum = post->GetMomentum();
    RecordExit(position.x() / m, position.y() / m, position.z() / m, momentum.x() / GeV,
               momentum.y() / GeV, momentum.z() / GeV);
  }

 private:
  const G4LogicalVolume* fTarget = nullptr;
};

class ActionInitialization final : public G4VUserActionInitialization {
 public:
  explicit ActionInitialization(double momentumGeV) : fMomentumGeV(momentumGeV) {}

  void Build() const override {
    SetUserAction(new PrimaryGenerator(fMomentumGeV));
    SetUserAction(new ExitRecorder);
  }

 private:
  double fMomentumGeV;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    G4cerr << "usage: yieldgen <n_events> <out.csv> [momentum_gev_c=26]" << G4endl;
    return 2;
  }
  const long nEvents = std::strtol(argv[1], nullptr, 10);
  const std::string outPath = argv[2];
  const double momentumGeV = (argc > 3) ? std::strtod(argv[3], nullptr) : 26.0;
  if (nEvents <= 0 || momentumGeV <= 0.0) {
    G4cerr << "n_events and momentum_gev_c must be positive" << G4endl;
    return 2;
  }
  if (nEvents > INT_MAX) {
    // BeamOn takes a G4int; a silent narrowing would make the header's
    // n_primaries disagree with the events actually simulated and bias
    // every weight derived from the table.
    G4cerr << "n_events must not exceed " << INT_MAX << G4endl;
    return 2;
  }

  g_out.open(outPath);
  if (!g_out) {
    G4cerr << "cannot open output file: " << outPath << G4endl;
    return 1;
  }
  g_out << "# latent-dirac antiproton yield table v1\n"
        << "# generator = engine/yieldgen\n"
        << "# geant4_version = " << G4Version << "\n"
        << "# physics_list = FTFP_BERT\n"
        << "# datasets = " << DatasetBasenames() << "\n"
        << "# patches = none\n"
        << "# primary = proton\n"
        << "# primary_momentum_gev_c = " << momentumGeV << "\n"
        << "# n_primaries = " << nEvents << "\n"
        << "# target = iridium cylinder r_mm=1.5 half_length_mm=27.5\n"
        << "# columns = x_m,y_m,z_m,px_gev_c,py_gev_c,pz_gev_c\n";

  auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);
  runManager->SetUserInitialization(new DetectorConstruction);
  runManager->SetUserInitialization(G4PhysListFactory().GetReferencePhysList("FTFP_BERT"));
  runManager->SetUserInitialization(new ActionInitialization(momentumGeV));
  runManager->Initialize();
  runManager->BeamOn(static_cast<G4int>(nEvents));

  delete runManager;

  // Trailing completion marker: the Python parser refuses tables without
  // it, so an interrupted run can never be replayed with biased weights.
  g_out << "# complete = true\n";
  g_out.close();
  return 0;
}
