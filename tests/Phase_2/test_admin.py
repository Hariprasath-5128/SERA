import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_backup_and_reset():
    # 1. Trigger Backup
    resp = client.post("/admin/backup", json={"version": "v1", "message": "Initial ground truth backup"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    import os
    assert os.path.join("backups", "v1") in data["backup_path"]
    
    # 2. Trigger Reset
    resp = client.post("/admin/reset", json={"commit_message": "Resetting for tests"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "pre_reset" in data["backup_path"]
