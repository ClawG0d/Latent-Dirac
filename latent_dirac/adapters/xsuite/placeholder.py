"""Placeholder interface for a future Xsuite adapter."""


class XsuiteAdapterPlaceholder:
    name = "Xsuite"
    available = False

    def describe(self) -> str:
        return "Xsuite integration is not implemented yet; this is an adapter placeholder."

    def run(self, *args, **kwargs):
        raise NotImplementedError(self.describe())
