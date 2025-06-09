import pytest
from lamanna.app import app as flask_app


@pytest.fixture()
def client():
    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as client:
        yield client

def test_model(client):
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
    print('####################################à S1')
    response = client.post("/infer", json=domanda)
    print('####################################à S2')
    assert response.status_code == 200
    print('####################################à S3')
    data = response.get_json()
    print('####################################à S4')
    print (data)
    print('####################################à S5')
    assert 'result' in data
    print('####################################à S6')
    result = data['result']
    print('####################################à S7')
    assert 'value' in result
    print('####################################à S8')
    assert result['value'] > 0