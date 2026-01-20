"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client for API."""
    from backend.api.main import app
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_check(self, client):
        """Test main health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "services" in data
    
    def test_readiness_probe(self, client):
        """Test readiness probe."""
        response = client.get("/health/ready")
        
        assert response.status_code == 200
        assert response.json()["ready"] is True
    
    def test_liveness_probe(self, client):
        """Test liveness probe."""
        response = client.get("/health/live")
        
        assert response.status_code == 200
        assert response.json()["alive"] is True


class TestExperimentsEndpoints:
    """Tests for experiments API."""
    
    def test_list_experiments_empty(self, client):
        """Test listing experiments when empty."""
        response = client.get("/api/experiments/")
        
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_create_experiment(self, client):
        """Test creating an experiment."""
        response = client.post(
            "/api/experiments/",
            json={
                "name": "test-experiment",
                "model_architecture": "whisper",
                "model_variant": "tiny",
                "batch_size": 8,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-experiment"
        assert data["status"] == "pending"
        assert "id" in data
    
    def test_get_nonexistent_experiment(self, client):
        """Test getting a nonexistent experiment."""
        response = client.get("/api/experiments/nonexistent-id")
        
        assert response.status_code == 404


class TestModelsEndpoints:
    """Tests for models API."""
    
    def test_list_architectures(self, client):
        """Test listing available architectures."""
        response = client.get("/api/models/architectures")
        
        assert response.status_code == 200
        architectures = response.json()
        assert len(architectures) > 0
        
        # Check whisper is available
        names = [a["name"] for a in architectures]
        assert "whisper" in names
    
    def test_get_architecture_details(self, client):
        """Test getting architecture details."""
        response = client.get("/api/models/architectures/whisper")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "whisper"
        assert "tiny" in data["variants"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
