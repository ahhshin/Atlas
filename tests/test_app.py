from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "page",
    [
        Path("app/Home.py"),
        Path("app/pages/1_World_State.py"),
        Path("app/pages/2_Experiments.py"),
        Path("app/pages/3_Data_Feeds.py"),
    ],
)
def test_streamlit_page_renders_without_exceptions(page: Path):
    app = AppTest.from_file(PROJECT_ROOT / page).run(timeout=30)

    assert not app.exception


def test_world_state_can_switch_to_explicit_synthetic_mode():
    app = AppTest.from_file(PROJECT_ROOT / "app/pages/1_World_State.py").run(timeout=30)
    app.segmented_control[0].set_value("Synthetic forecast").run(timeout=30)

    assert not app.exception
    assert "SYNTHETIC" in app.caption[0].value
    assert app.selectbox[1].label == "Model"
