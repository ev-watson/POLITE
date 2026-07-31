import pytest

from obs_utils.roi import CameraROI, EffectiveArea, apply_roi


class FakeCamera:
    CameraXSize = 6280
    CameraYSize = 4210

    def __init__(self):
        self.BinX = self.BinY = 1
        self.StartX = self.StartY = 0
        self.NumX, self.NumY = self.CameraXSize, self.CameraYSize


def test_provisional_science_roi_is_valid_for_the_raw_sensor():
    roi = CameraROI(startx=500, starty=0, numx=5280, numy=4210)
    assert roi.validate(6280, 4210) == roi


def test_effective_area_rejects_an_roi_that_touches_overscan():
    roi = CameraROI(startx=500, starty=0, numx=5280, numy=4210)
    with pytest.raises(ValueError, match="effective area"):
        roi.validate(6280, 4210, effective_area=EffectiveArea(501, 0, 5278, 4210))


def test_apply_roi_requires_exact_readback():
    camera = FakeCamera()
    roi = CameraROI(startx=500, starty=0, numx=5280, numy=4210)
    assert apply_roi(camera, roi) == roi


def test_night_plan_preserves_horizontal_roi_origin():
    from obs_utils.night_plan import _capture_kwargs

    assert _capture_kwargs({"startx": 500, "starty": 0, "numx": 5280, "numy": 4210}) == {
        "startx": 500, "starty": 0, "numx": 5280, "numy": 4210,
    }
