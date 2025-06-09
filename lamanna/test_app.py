import pytest
from app import app as flask_app

#CONSUMER

domanda = {
  "name": "Clair Obscur: Expedition 33",
  "pos_perc": 0.952213,
  "release_date": 2025.0424,
  "publisher": "Annapurna Interactive",
  "median_playtime": 60,
  "price": 49.99,
  "Genre": {
    "Action": 1,
    "Adventure": 0,
    "Casual": 0,
    "Early Access": 0,
    "Free to Play": 0,
    "Indie": 1,
    "Massively Multiplayer": 0,
    "RPG": 1,
    "Racing": 0,
    "Simulation": 0,
    "Sports": 0,
    "Strategy": 0
  }
}

@pytest.fixture()
def client():
    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as client:
        yield client

def test_model(client):
    response = client.post("/infer", jso=domanda)
    assert response.status_code == 200
    data = response.get_json()
    print (data)
    assert data == {"message": "Hello Alessandro!"}