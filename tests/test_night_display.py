from obs_utils.night_display import NightReporter, estimated_duration_s


class _Frame:
    def __init__(self, exposure_s, count):
        self.exposure_s = exposure_s
        self.count = count


def test_estimated_duration_weights_multi_minute_frames():
    plans = [_Frame(0.0, 25), _Frame(5.0, 5), _Frame(60.0, 5), _Frame(150.0, 5)]

    assert estimated_duration_s(plans) == 1175.0


def test_reporter_eta_decrements_by_exposure_not_average_frame_rate():
    reporter = NightReporter(4, estimated_duration_s=370.0, enabled=False)

    reporter.frame_captured("BIAS", exp=0.0)

    assert reporter._remaining_estimate_s == 367.5
    reporter.frame_captured("DARK", exp=150.0)
    assert reporter._remaining_estimate_s == 215.0
