from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_expected_files_exist():
    assert (ROOT / "scripts" / "cyclone_tracker.py").exists()
    assert (ROOT / ".github" / "workflows" / "cyclone_update.yml").exists()
    assert (ROOT / "docs" / "index.html").exists()
    assert (ROOT / "config" / "settings.yml").exists()
