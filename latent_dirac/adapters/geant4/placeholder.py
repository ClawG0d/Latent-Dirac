"""Placeholder interface for a future Geant4 adapter."""


class Geant4AdapterPlaceholder:
    name = "Geant4"
    available = False

    def describe(self) -> str:
        return "Geant4 integration is not implemented yet; this is an adapter placeholder."

    def run(self, *args, **kwargs):
        raise NotImplementedError(self.describe())
