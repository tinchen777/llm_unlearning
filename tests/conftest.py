def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: runs the real LUNAR / NEAR scripts end to end on a tiny local model (minutes); "
                                       "deselect with -m 'not smoke'")
