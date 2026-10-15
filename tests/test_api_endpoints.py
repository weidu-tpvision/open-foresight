"""Integration tests for FastAPI endpoints."""

import pytest
from starlette.testclient import TestClient
from daemon.app import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_studio_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "OpenForesight" in response.text


def test_favicon_endpoint(client):
    response = client.get("/favicon.ico")
    assert response.status_code == 204


def test_health_check_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Linux" in data["platform"]
    assert data["cuda_available"] is True
    assert len(data["device"]) > 0


def test_audio_devices_discovery(client):
    response = client.get("/api/audio/devices")
    assert response.status_code == 200
    data = response.json()
    assert "devices" in data
    # At least the USB Audio or HDMI Sink should be visible
    device_names = [d["name"] for d in data["devices"]]
    assert any("alsa" in name for name in device_names)


def test_session_lifecycle(client):
    # 1. Start session
    start_resp = client.post(
        "/api/sessions/start",
        json={"title": "Sprint Planning Test"},
    )
    assert start_resp.status_code == 200
    session_data = start_resp.json()
    session_id = session_data["session_id"]
    assert session_data["status"] == "recording"
    assert session_data["title"] == "Sprint Planning Test"

    # 2. Get list of sessions
    list_resp = client.get("/api/sessions")
    assert list_resp.status_code == 200
    sessions = list_resp.json()["sessions"]
    assert any(s["id"] == session_id for s in sessions)

    # 3. Stop session
    stop_resp = client.post(f"/api/sessions/{session_id}/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["status"] == "stopped"

    # 4. Verify session detail shows stopped
    detail_resp = client.get(f"/api/sessions/{session_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["session"]["status"] == "stopped"


if __name__ == "__main__":
    with TestClient(app) as c:
        test_health_check_endpoint(c)
        test_audio_devices_discovery(c)
        test_session_lifecycle(c)
        print("All API integration tests passed successfully!")
