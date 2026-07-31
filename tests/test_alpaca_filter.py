from obs_utils.alpaca import set_filter_position


class FakeWheel:
    Names = ["Clear", "B", "V", "R", "Dark"]
    IsMoving = False

    def __init__(self):
        self.Position = 0


def test_filter_label_accepts_underscore_space_and_case_variants():
    wheel = FakeWheel()
    landed = set_filter_position(
        wheel, "Photometric_V", names_override=["Clear", "Photometric B", "Photometric V", "Photometric R", "Dark"],
        poll_s=0.0,
    )
    assert landed.fwpos == 2
    assert landed.filtrdy
