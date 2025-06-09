import pytest
from app import app as flask_app

#CONSUMER

to_predict_CSV = "name,pos_perc,release_date,publisher,median_playtime,price,Genre: Action,Genre: Adventure,Genre: Casual,Genre: Early Access,Genre: Free to Play,Genre: Indie,Genre: Massively Multiplayer,Genre: RPG,Genre: Racing,Genre: Simulation,Genre: Sports,Genre: Strategy\nClair Obscur: Expedition 33,0.952213,2025.0424,Annapurna Interactive,60,49.99,1,0,0,0,0,1,0,1,0,0,0,0"

@pytest.fixture()
def client():
    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as client:
        yield client

def test_hello(client):
    response = client.post("/infer", json={"name": "Alessandro"})
    assert response.status_code == 200
    data = response.get_json()
    assert data == {"message": "Hello Alessandro!"}