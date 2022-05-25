import json
import os
from fastapi.testclient import TestClient
from fastapi import status
from main import app


client = TestClient(app)

os.environ['ACCESS_TOKEN'] = 'access_token'


def test_unauthorized_request():
    response = client.post(
        url='/predict',
        json={'pixels': 'Random pixels'}
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_invalid_access_token_request():
    response = client.post(
        url='/predict',
        headers={'Authorization': 'Bearer invalid_token'},
        json={'pixels': 'Random pixels'}
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_invalid_encoded_pixels_request():
    response = client.post(
        url='/predict',
        headers={'Authorization': 'Bearer access_token'},
        json={'pixels': 'Random pixels'}
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_invalid_image_request():
    response = client.post(
        url='/predict',
        headers={'Authorization': 'Bearer access_token'},
        json={'pixels': 'aGVsbG8gd29ybGQh'}
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_valid_request():
    with open('data/samples/encoded/0427_head_203.txt') as f:
        encoded_pixels = f.read()
    response = client.post(
        url='/predict',
        headers={'Authorization': 'Bearer access_token'},
        json={'pixels': encoded_pixels}
    )
    res_body = json.loads(response.content.decode())

    assert response.status_code == status.HTTP_200_OK
    assert res_body.get('body_part') == 'head'
    assert res_body.get('image_bytes') is not None
