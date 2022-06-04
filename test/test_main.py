import json
import os

from fastapi.testclient import TestClient
from fastapi import status
from main import app, ACCESS_TOKEN_ENV_KEY

predict_url = '/predict'
access_token = 'access_token'
client = TestClient(app)

os.environ[ACCESS_TOKEN_ENV_KEY] = access_token


def test_unauthorized_request():
    resp = client.post(
        url=predict_url,
        json={'data': 'Random pixels'}
    )

    assert resp.status_code == status.HTTP_403_FORBIDDEN


def test_invalid_access_token_request():
    resp = client.post(
        url=predict_url,
        headers={'Authorization': 'Bearer invalid_token'},
        json={'data': 'Random pixels'}
    )

    assert resp.status_code == status.HTTP_403_FORBIDDEN


def test_invalid_encoded_pixels_request():
    resp = client.post(
        url=predict_url,
        headers={'Authorization': f'Bearer {access_token}'},
        json={'data': 'Random pixels'}
    )

    assert resp.status_code == status.HTTP_400_BAD_REQUEST


def test_invalid_image_request():
    resp = client.post(
        url=predict_url,
        headers={'Authorization': f'Bearer {access_token}'},
        json={'data': 'aGVsbG8gd29ybGQh'}
    )

    assert resp.status_code == status.HTTP_400_BAD_REQUEST


def test_valid_request():
    with open('test/data/samples/encoded/0427_head_203.txt') as f:
        encoded_pixels = f.read()

    resp = client.post(
        url=predict_url,
        headers={'Authorization': f'Bearer {access_token}'},
        json={'data': encoded_pixels}
    )
    resp_body = json.loads(resp.content.decode())

    assert resp.status_code == status.HTTP_200_OK
    assert resp_body.get('prediction') is not None
    assert resp_body.get('prediction').get('body_part') == 'head'
    assert resp_body.get('photo') is not None
