"""Placeholder interface for a future ROOT adapter."""


class RootAdapterPlaceholder:
    name = "ROOT"
    available = False

    def describe(self) -> str:
        return "ROOT integration is not implemented yet; this is an adapter placeholder."

    def run(self, *args, **kwargs):
        raise NotImplementedError(self.describe())
