// transformer: matter-transform engine application (M2, Matter component).
//
// Reads a latent-dirac phase-space CSV, fires each row as one primary
// through a slab of a NIST material with the FTFP_BERT reference physics
// list, and records the primary's phase space where it crosses a scoring
// plane 1 mm downstream of the slab. Absorbed, annihilated, stopped, or
// backscattered primaries produce no output row. Contract:
// docs/superpowers/specs/2026-07-05-geant4-matter-adapter-design.md
//
// Usage: transformer <input.csv> <output.csv> <nist_material> <thickness_mm>

#include <climits>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "G4Box.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
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
#include "G4UserSteppingAction.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4Version.hh"

namespace {

constexpr double kHalfWidth = 20.0 * cm;
constexpr double kPlaneGap = 1.0 * mm;
constexpr double kPlaneHalfThickness = 0.05 * mm;
constexpr double kWorldHalfLength = 0.6 * m;  // keep in sync with the Python adapter

struct Row {
  long id;
  double x_m, y_m, z_m;
  double px_gev, py_gev, pz_gev;
};

std::vector<Row> g_rows;
std::string g_speciesName;
double g_thickness = 0.0;
std::string g_material;

std::ofstream g_out;
std::mutex g_outMutex;

void RecordExit(long id, double xm, double ym, double zm, double pxg, double pyg, double pzg) {
  std::lock_guard<std::mutex> lock(g_outMutex);
  g_out << id << ',' << xm << ',' << ym << ',' << zm << ',' << pxg << ',' << pyg << ',' << pzg
        << '\n';
}

std::string DatasetBasenames() {
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

bool ParseInput(const std::string& path, std::string& error) {
  std::ifstream in(path);
  if (!in) {
    error = "cannot open input file: " + path;
    return false;
  }
  bool complete = false;
  std::string line;
  long lineNo = 0;
  while (std::getline(in, line)) {
    ++lineNo;
    if (line.empty()) continue;
    if (line[0] == '#') {
      const auto eq = line.find('=');
      if (eq != std::string::npos) {
        auto trim = [](std::string s) {
          const auto a = s.find_first_not_of(" \t#");
          const auto b = s.find_last_not_of(" \t\r");
          return (a == std::string::npos) ? std::string() : s.substr(a, b - a + 1);
        };
        const std::string key = trim(line.substr(0, eq));
        const std::string value = trim(line.substr(eq + 1));
        if (key == "species") g_speciesName = value;
        if (key == "complete" && value == "true") complete = true;
      }
      continue;
    }
    Row row{};
    char comma;
    std::istringstream stream(line);
    if (!(stream >> row.id >> comma >> row.x_m >> comma >> row.y_m >> comma >> row.z_m >> comma >>
          row.px_gev >> comma >> row.py_gev >> comma >> row.pz_gev)) {
      error = "malformed row at line " + std::to_string(lineNo);
      return false;
    }
    g_rows.push_back(row);
  }
  if (g_speciesName.empty()) {
    error = "input is missing the '# species = <name>' header";
    return false;
  }
  if (!complete) {
    error = "input is missing the trailing '# complete = true' marker";
    return false;
  }
  if (g_rows.empty()) {
    error = "input contains no phase-space rows";
    return false;
  }
  if (static_cast<long long>(g_rows.size()) > INT_MAX) {
    error = "too many rows for a single run";
    return false;
  }
  return true;
}

class DetectorConstruction final : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    auto* vacuum = nist->FindOrBuildMaterial("G4_Galactic");
    auto* material = nist->FindOrBuildMaterial(g_material);
    if (material == nullptr) {
      G4Exception("transformer", "mat001", FatalException,
                  ("unknown NIST material: " + g_material).c_str());
    }

    auto* worldSolid = new G4Box("world", kHalfWidth + 5.0 * cm, kHalfWidth + 5.0 * cm,
                                 kWorldHalfLength);
    auto* worldLV = new G4LogicalVolume(worldSolid, vacuum, "world");
    auto* worldPV = new G4PVPlacement(nullptr, {}, worldLV, "world", nullptr, false, 0);

    auto* slabSolid = new G4Box("slab", kHalfWidth, kHalfWidth, 0.5 * g_thickness);
    auto* slabLV = new G4LogicalVolume(slabSolid, material, "slab");
    new G4PVPlacement(nullptr, {0.0, 0.0, 0.5 * g_thickness}, slabLV, "slab", worldLV, false, 0);

    auto* planeSolid = new G4Box("plane", kHalfWidth, kHalfWidth, kPlaneHalfThickness);
    auto* planeLV = new G4LogicalVolume(planeSolid, vacuum, "plane");
    new G4PVPlacement(nullptr, {0.0, 0.0, g_thickness + kPlaneGap}, planeLV, "plane", worldLV,
                      false, 0);

    return worldPV;
  }
};

class PrimaryGenerator final : public G4VUserPrimaryGeneratorAction {
 public:
  PrimaryGenerator() {
    fGun.SetParticleDefinition(G4ParticleTable::GetParticleTable()->FindParticle(g_speciesName));
  }

  void GeneratePrimaries(G4Event* event) override {
    const Row& row = g_rows[static_cast<size_t>(event->GetEventID())];
    fGun.SetParticlePosition({row.x_m * m, row.y_m * m, row.z_m * m});
    const G4ThreeVector momentum(row.px_gev * GeV, row.py_gev * GeV, row.pz_gev * GeV);
    fGun.SetParticleMomentumDirection(momentum.unit());
    fGun.SetParticleMomentum(momentum.mag());
    fGun.GeneratePrimaryVertex(event);
  }

 private:
  G4ParticleGun fGun;
};

class PlaneRecorder final : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    const auto* track = step->GetTrack();
    if (track->GetParentID() != 0) return;

    const auto* post = step->GetPostStepPoint();
    const auto* postVolume = post->GetPhysicalVolume();
    if (postVolume == nullptr) return;
    if (fPlane == nullptr) {
      fPlane = G4LogicalVolumeStore::GetInstance()->GetVolume("plane");
    }
    const bool entersPlane =
        postVolume->GetLogicalVolume() == fPlane &&
        step->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume() != fPlane;
    if (!entersPlane) return;

    const auto eventId =
        G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
    const Row& row = g_rows[static_cast<size_t>(eventId)];
    const auto position = post->GetPosition();
    const auto momentum = post->GetMomentum();
    RecordExit(row.id, position.x() / m, position.y() / m, position.z() / m, momentum.x() / GeV,
               momentum.y() / GeV, momentum.z() / GeV);
  }

 private:
  const G4LogicalVolume* fPlane = nullptr;
};

class ActionInitialization final : public G4VUserActionInitialization {
 public:
  void Build() const override {
    SetUserAction(new PrimaryGenerator);
    SetUserAction(new PlaneRecorder);
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    G4cerr << "usage: transformer <input.csv> <output.csv> <nist_material> <thickness_mm>"
           << G4endl;
    return 2;
  }
  const std::string inPath = argv[1];
  const std::string outPath = argv[2];
  g_material = argv[3];
  const double thicknessMm = std::strtod(argv[4], nullptr);
  if (thicknessMm <= 0.0) {
    G4cerr << "thickness_mm must be positive" << G4endl;
    return 2;
  }
  g_thickness = thicknessMm * mm;
  if (g_thickness + kPlaneGap + kPlaneHalfThickness >= kWorldHalfLength) {
    G4cerr << "transformer: thickness_mm too large for the engine world envelope ("
           << kWorldHalfLength / mm << " mm)" << G4endl;
    return 2;
  }

  std::string error;
  if (!ParseInput(inPath, error)) {
    G4cerr << "transformer: " << error << G4endl;
    return 2;
  }

  // Fail fast on an unsupported species BEFORE the run manager builds the
  // primary generator: in a serial build ActionInitialization::Build() runs
  // inside SetUserInitialization and would hand a null particle to the gun.
  static const std::set<std::string> kSupported = {"e-", "e+", "proton", "anti_proton"};
  if (kSupported.find(g_speciesName) == kSupported.end()) {
    G4cerr << "transformer: unsupported species '" << g_speciesName << "'" << G4endl;
    return 2;
  }

  g_out.open(outPath);
  if (!g_out) {
    G4cerr << "cannot open output file: " << outPath << G4endl;
    return 1;
  }
  g_out << std::setprecision(9);  // match the %.9g the Python side writes with
  g_out << "# latent-dirac phase space v1\n"
        << "# species = " << g_speciesName << "\n"
        << "# generator = engine/transformer\n"
        << "# geant4_version = " << G4Version << "\n"
        << "# physics_list = FTFP_BERT\n"
        << "# datasets = " << DatasetBasenames() << "\n"
        << "# patches = none\n"
        << "# material = " << g_material << "\n"
        << "# thickness_mm = " << thicknessMm << "\n"
        << "# n_primaries = " << g_rows.size() << "\n"
        << "# columns = id,x_m,y_m,z_m,px_gev_c,py_gev_c,pz_gev_c\n";

  auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);
  runManager->SetUserInitialization(new DetectorConstruction);
  runManager->SetUserInitialization(G4PhysListFactory().GetReferencePhysList("FTFP_BERT"));
  runManager->SetUserInitialization(new ActionInitialization);
  runManager->Initialize();
  runManager->BeamOn(static_cast<G4int>(g_rows.size()));
  delete runManager;

  g_out << "# complete = true\n";
  g_out.close();
  return 0;
}
